use crate::protocol::{ErrorCode, Reply, RpcError, ServerEvent};
use crate::server::Daemon;

pub(crate) fn get_config(d: &Daemon) -> Result<Reply, RpcError> {
    Ok(Reply::Config(d.runtime.config()))
}

/// Every catalog model. Clients derive their own views from the full set, such
/// as the picker's judge-capable list and the auth modal's provider list.
pub(crate) fn list_models(d: &Daemon) -> Result<Reply, RpcError> {
    Ok(Reply::Models(d.runtime.catalog().iter().cloned().collect()))
}

pub(crate) fn set_model_role(
    d: &Daemon,
    role: String,
    selector: String,
) -> Result<Reply, RpcError> {
    d.runtime.set_model_role(&role, &selector).map_err(|err| {
        RpcError::new(
            ErrorCode::Invalid,
            format!("could not assign @{role}: {err:#}"),
        )
    })?;
    let config = d.runtime.config();
    d.emit(ServerEvent::Config {
        config: config.clone(),
    });
    Ok(Reply::Config(config))
}
