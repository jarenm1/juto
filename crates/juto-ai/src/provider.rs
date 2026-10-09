//! HTTP streaming transports, independent of agent execution and persistence.
//! Wire behavior references OMP 579da1d6; see licenses/OMP-MIT.txt.
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use juto_catalog::Model;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::sse::{NdjsonDecoder, SseDecoder};
use crate::{
    Context, Credential, CredentialStore, Provider, ProviderError, ProviderEvent, ProviderStream,
    StreamOptions,
};

mod decode;
mod requests;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Api {
    Chat,
    Responses,
    Codex,
    Azure,
    Anthropic,
    Gemini,
    Ollama,
}
impl Api {
    fn for_model(model: &Model) -> Result<Self, ProviderError> {
        match model.api.as_str() {
            "openai-completions" | "openrouter" => Ok(Self::Chat),
            "openai-responses" => Ok(Self::Responses),
            "openai-codex-responses" => Ok(Self::Codex),
            "azure-openai-responses" => Ok(Self::Azure),
            "anthropic-messages" => Ok(Self::Anthropic),
            "google-generative-ai" => Ok(Self::Gemini),
            "ollama-chat" => Ok(Self::Ollama),
            _ => Err(ProviderError::Unsupported(model.api.clone())),
        }
    }
}

pub struct HttpProvider {
    credentials: Arc<CredentialStore>,
    client: reqwest::Client,
}
impl HttpProvider {
    pub fn new(credentials: Arc<CredentialStore>) -> Result<Self, ProviderError> {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(transport_error)?;
        Ok(Self {
            credentials,
            client,
        })
    }
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }
}

#[async_trait]
impl Provider for HttpProvider {
    async fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
        cancel: CancellationToken,
    ) -> Result<ProviderStream, ProviderError> {
        let api = Api::for_model(model)?;
        let credential = tokio::select! {
            _=cancel.cancelled()=>return Err(ProviderError::Aborted),
            result=self.credentials.resolve(model)=>result.map_err(|error|ProviderError::Authentication(error.to_string()))?,
        };
        let request = requests::build(
            &self.client,
            api,
            model,
            context,
            options,
            credential.as_ref(),
        )?;
        let response = tokio::select! {
            _=cancel.cancelled()=>return Err(ProviderError::Aborted),
            result=tokio::time::timeout(options.timeout,request.send())=>result.map_err(|_|ProviderError::Transport("request timeout".into()))?.map_err(transport_error)?,
        };
        if !response.status().is_success() {
            return Err(tokio::select! {
                _=cancel.cancelled()=>ProviderError::Aborted,
                result=tokio::time::timeout(options.timeout,http_error(response,credential.as_ref()))=>result.unwrap_or_else(|_|ProviderError::Transport("error response timeout".into())),
            });
        }
        let model = model.clone();
        let timeout = options.timeout;
        Ok(Box::pin(async_stream::try_stream! {
            yield ProviderEvent::Start;
            let mut body=response.bytes_stream();
            let mut sse=SseDecoder::new();let mut ndjson=NdjsonDecoder::new();
            let mut decoder=decode::Decoder::new(api,&model);
            let mut pending=Vec::new();let mut terminal=false;
            while !terminal {
                let chunk=tokio::select! {
                    _=cancel.cancelled()=>Err(ProviderError::Aborted),
                    result=tokio::time::timeout(timeout,body.next())=>result.map_err(|_|ProviderError::Transport("stream idle timeout".into())),
                }?;
                let Some(chunk)=chunk else {break};let chunk=chunk.map_err(transport_error)?;
                if api==Api::Ollama {
                    for payload in ndjson.feed(&chunk)? {
                        terminal=decoder.feed(&payload,&mut pending)?;
                        for event in pending.drain(..) {yield event;}
                        if terminal {break;}
                    }
                } else {
                    for frame in sse.feed(&chunk)? {
                        if frame.data.is_empty() {continue;}
                        terminal=decoder.feed(&frame.data,&mut pending)?;
                        for event in pending.drain(..) {yield event;}
                        if terminal {break;}
                    }
                }
            }
            if !terminal {
                if api==Api::Ollama {
                    for payload in ndjson.finish()? {decoder.feed(&payload,&mut pending)?;}
                } else {
                    for frame in sse.finish()? {if !frame.data.is_empty() {decoder.feed(&frame.data,&mut pending)?;}}
                }
                for event in pending.drain(..) {yield event;}
            }
            decoder.close_blocks(&mut pending);
            for event in pending.drain(..) {yield event;}
            let message=decoder.finish(&model)?;
            yield ProviderEvent::Done {message};
        }))
    }
}

fn transport_error(error: reqwest::Error) -> ProviderError {
    ProviderError::Transport(error.without_url().to_string())
}
async fn http_error(response: reqwest::Response, credential: Option<&Credential>) -> ProviderError {
    let status = response.status().as_u16();
    let retry_after = response
        .headers()
        .get("retry-after")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse().ok());
    let mut body = response.bytes_stream();
    let mut bytes = Vec::new();
    while let Some(Ok(chunk)) = body.next().await {
        let retain = chunk.len().min(32_768 - usize::min(bytes.len(), 32_768));
        bytes.extend_from_slice(&chunk[..retain]);
        if bytes.len() == 32_768 {
            break;
        }
    }
    let parsed = serde_json::from_slice::<Value>(&bytes).ok();
    let message = parsed
        .as_ref()
        .and_then(|body| {
            body.pointer("/error/message")
                .or_else(|| body.get("message"))
        })
        .and_then(Value::as_str);
    let mut message = message
        .map(str::to_owned)
        .unwrap_or_else(|| String::from_utf8_lossy(&bytes).into_owned());
    if let Some(credential) = credential {
        if !credential.expose_secret().is_empty() {
            message = message.replace(credential.expose_secret(), "<redacted>");
        }
    }
    ProviderError::Http {
        status,
        message,
        retry_after,
    }
}
