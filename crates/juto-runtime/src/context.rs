//! Project instruction and skill discovery.
//!
//! Behavioral reference: oh-my-pi 579da1d6 context files (`.omp/AGENTS.md`
//! ancestor walk + user AGENTS.md) and skill providers
//! (`<root>/<name>/SKILL.md` with YAML frontmatter).
//!
//! Juto's native locations: `AGENTS.md`, `.juto/AGENTS.md`, `.juto/skills/`,
//! plus the user's global dir. `.omp` equivalents are read for compatibility
//! when present. Instruction files are loaded once each, ordered global →
//! filesystem root → cwd, so broad rules precede local ones. `@`-imports are
//! not followed: only the declared files participate, keeping each project's
//! context isolated.

use std::{
    collections::BTreeMap,
    io,
    path::{Path, PathBuf},
};

use serde::Deserialize;

#[derive(Debug, thiserror::Error)]
pub enum ContextError {
    #[error("failed to read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("malformed frontmatter in {path}: {reason}")]
    Frontmatter { path: PathBuf, reason: String },
}

pub type Result<T, E = ContextError> = std::result::Result<T, E>;

/// Instruction file names checked in each ancestor directory, in order.
const DIR_INSTRUCTION_FILES: &[&str] = &["AGENTS.md", ".juto/AGENTS.md", ".omp/AGENTS.md"];

/// A discovered skill: frontmatter metadata plus the real file path a host
/// hands to tools — no `skill://` indirection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    /// Path to the SKILL.md file itself.
    pub path: PathBuf,
}

#[derive(Debug, Deserialize)]
struct SkillFrontmatter {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    description: Option<String>,
}

fn io_err(path: &Path) -> impl FnOnce(io::Error) -> ContextError + '_ {
    move |source| ContextError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Ancestors of `cwd` ordered filesystem-root → cwd.
fn ancestors_root_first(cwd: &Path) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = cwd.ancestors().map(Path::to_path_buf).collect();
    dirs.reverse();
    dirs
}

/// Load instruction files: `<global_dir>/AGENTS.md` first, then for every
/// directory from the filesystem root down to `cwd` the files `AGENTS.md`,
/// `.juto/AGENTS.md`, and `.omp/AGENTS.md`. Each file is included once —
/// a `global_dir` that coincides with an ancestor does not double-load.
///
/// Returns file contents in load order. Missing/unreadable-as-missing files
/// are skipped; genuine read errors (permissions, I/O) propagate.
pub fn load_instructions(cwd: &Path, global_dir: &Path) -> Result<Vec<String>> {
    let mut files: Vec<PathBuf> = vec![global_dir.join("AGENTS.md")];
    for dir in ancestors_root_first(cwd) {
        for name in DIR_INSTRUCTION_FILES {
            files.push(dir.join(name));
        }
    }
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for path in files {
        // Dedup on the lexical path; canonicalize only when the file exists
        // so symlinks that converge also dedup.
        let key = path.canonicalize().unwrap_or_else(|_| path.clone());
        if !seen.insert(key) {
            continue;
        }
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                if !text.trim().is_empty() {
                    out.push(text);
                }
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(io_err(&path)(e)),
        }
    }
    Ok(out)
}

/// Parse a SKILL.md body: `---` frontmatter block then free text.
fn parse_skill(path: &Path, text: &str) -> Result<Option<(Option<String>, Option<String>)>> {
    let Some((yaml, _body)) = split_frontmatter(path, text)? else {
        return Ok(None);
    };
    let fm: SkillFrontmatter =
        serde_yaml::from_str(yaml).map_err(|e| ContextError::Frontmatter {
            path: path.to_path_buf(),
            reason: e.to_string(),
        })?;
    Ok(Some((fm.name, fm.description)))
}

/// Split a markdown file into its `---` frontmatter YAML and its body. Returns
/// `None` when the file has no frontmatter block.
pub(crate) fn split_frontmatter<'a>(
    path: &Path,
    text: &'a str,
) -> Result<Option<(&'a str, &'a str)>> {
    let trimmed = text.trim_start_matches('\u{feff}');
    if !trimmed.starts_with("---") {
        return Ok(None);
    }
    let rest = &trimmed[3..];
    // Frontmatter must start on its own line.
    if !rest.starts_with(['\n', '\r']) {
        return Ok(None);
    }
    let rest = rest.trim_start_matches(['\r', '\n']);
    let end = rest
        .find("\n---")
        .ok_or_else(|| ContextError::Frontmatter {
            path: path.to_path_buf(),
            reason: "opening '---' has no closing delimiter".into(),
        })?;
    // The body starts on the line after the closing delimiter.
    let after = &rest[end + "\n---".len()..];
    let body = after.split_once('\n').map_or("", |(_, body)| body);
    Ok(Some((&rest[..end], body)))
}

fn scan_skills_root(root: &Path, out: &mut BTreeMap<String, Skill>) -> Result<()> {
    let entries = match std::fs::read_dir(root) {
        Ok(entries) => entries,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(io_err(root)(e)),
    };
    for entry in entries {
        let entry = entry.map_err(io_err(root))?;
        let dir = entry.path();
        let is_dir = entry.file_type().map_err(io_err(&dir))?.is_dir();
        if !is_dir {
            continue;
        }
        let skill_md = dir.join("SKILL.md");
        let text = match std::fs::read_to_string(&skill_md) {
            Ok(text) => text,
            Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
            Err(e) => return Err(io_err(&skill_md)(e)),
        };
        let Some((name, description)) = parse_skill(&skill_md, &text)? else {
            continue;
        };
        let name = name
            .filter(|n| !n.trim().is_empty())
            .unwrap_or_else(|| entry.file_name().to_string_lossy().into_owned());
        out.entry(name.clone()).or_insert(Skill {
            name,
            description: description.unwrap_or_default(),
            path: skill_md,
        });
    }
    Ok(())
}

/// Discover project skills first, then user skills from `<global_dir>/skills/`
/// and `<global_dir>/.omp/skills/`, then bundled harness skills. Nearer-to-cwd
/// project skills and user-installed global skills shadow same-named defaults.
pub fn discover_skills(cwd: &Path, global_dir: &Path) -> Result<Vec<Skill>> {
    let mut skills = BTreeMap::new();
    for dir in cwd.ancestors() {
        scan_skills_root(&dir.join(".juto/skills"), &mut skills)?;
        scan_skills_root(&dir.join(".omp/skills"), &mut skills)?;
    }
    scan_skills_root(&global_dir.join("skills"), &mut skills)?;
    scan_skills_root(&global_dir.join(".omp/skills"), &mut skills)?;
    scan_skills_root(&global_dir.join("bundled-skills"), &mut skills)?;
    Ok(skills.into_values().collect())
}

/// Install Juto's bundled repository-setup skill in the runtime data directory.
/// The complete file is linked into place atomically; existing user content wins.
pub fn install_bundled_skills(global_dir: &Path) -> Result<()> {
    use std::{
        io::Write as _,
        sync::atomic::{AtomicU64, Ordering},
    };

    static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    let directory = global_dir.join("bundled-skills/juto-repo-setup");
    std::fs::create_dir_all(&directory).map_err(io_err(&directory))?;
    let path = directory.join("SKILL.md");
    let (temporary, mut file) = loop {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temporary =
            directory.join(format!(".SKILL.md.{}.{}.tmp", std::process::id(), sequence));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(file) => break (temporary, file),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(io_err(&temporary)(error)),
        }
    };
    let write_result = file
        .write_all(include_bytes!("../skills/juto-repo-setup/SKILL.md"))
        .and_then(|()| file.sync_all());
    drop(file);
    if let Err(error) = write_result {
        let _ = std::fs::remove_file(&temporary);
        return Err(io_err(&temporary)(error));
    }

    let link_result = std::fs::hard_link(&temporary, &path);
    let cleanup_result = std::fs::remove_file(&temporary);
    match link_result {
        Ok(()) => cleanup_result.map_err(io_err(&temporary))?,
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            cleanup_result.map_err(io_err(&temporary))?;
        }
        Err(error) => {
            let _ = cleanup_result;
            return Err(io_err(&path)(error));
        }
    }
    Ok(())
}

#[cfg(test)]
mod bundled_skill_tests {
    use super::*;

    #[test]
    fn installs_bundled_skill_for_global_discovery_without_overwriting_user_edits() {
        let tmp = tempfile::tempdir().unwrap();
        let global = tmp.path().join("global");
        install_bundled_skills(&global).unwrap();

        let skills = discover_skills(tmp.path(), &global).unwrap();
        let skill = skills
            .iter()
            .find(|skill| skill.name == "juto-repo-setup")
            .unwrap();
        assert!(skill.description.contains("target repository"));
        assert!(
            std::fs::read_to_string(&skill.path)
                .unwrap()
                .contains("## Completion criteria")
        );

        std::fs::write(&skill.path, "user version").unwrap();
        install_bundled_skills(&global).unwrap();
        assert_eq!(
            std::fs::read_to_string(&skill.path).unwrap(),
            "user version"
        );
    }

    #[test]
    fn global_compatibility_skill_shadows_bundled_skill() {
        let tmp = tempfile::tempdir().unwrap();
        let global = tmp.path().join("global");
        let custom = global.join(".omp/skills/juto-repo-setup");
        std::fs::create_dir_all(&custom).unwrap();
        std::fs::write(
            custom.join("SKILL.md"),
            "---\nname: juto-repo-setup\ndescription: user version\n---\ncustom\n",
        )
        .unwrap();

        install_bundled_skills(&global).unwrap();

        let skill = discover_skills(tmp.path(), &global)
            .unwrap()
            .into_iter()
            .find(|skill| skill.name == "juto-repo-setup")
            .unwrap();
        assert_eq!(skill.description, "user version");
        assert_eq!(
            std::fs::read_to_string(skill.path).unwrap().lines().last(),
            Some("custom")
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree() -> (tempfile::TempDir, PathBuf, PathBuf) {
        // root/project/subdir layout; returns (tempdir, cwd, global_dir)
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let cwd = root.join("project/subdir");
        std::fs::create_dir_all(&cwd).unwrap();
        let global = root.join("globalhome");
        std::fs::create_dir_all(&global).unwrap();
        (tmp, cwd, global)
    }

    #[test]
    fn instructions_order_and_once() {
        let (tmp, cwd, global) = tree();
        let root = tmp.path();
        std::fs::write(global.join("AGENTS.md"), "GLOBAL").unwrap();
        std::fs::write(root.join("AGENTS.md"), "ROOT").unwrap();
        let proj = root.join("project");
        std::fs::write(proj.join("AGENTS.md"), "PROJ").unwrap();
        std::fs::create_dir_all(proj.join(".juto")).unwrap();
        std::fs::write(proj.join(".juto/AGENTS.md"), "PROJ_JUTO").unwrap();
        std::fs::create_dir_all(cwd.join(".omp")).unwrap();
        std::fs::write(cwd.join(".omp/AGENTS.md"), "CWD_OMP").unwrap();

        let got = load_instructions(&cwd, &global).unwrap();
        assert_eq!(
            got,
            vec!["GLOBAL", "ROOT", "PROJ", "PROJ_JUTO", "CWD_OMP"],
            "global → root → cwd, each file once"
        );
    }

    #[test]
    fn instructions_missing_ok_unreadable_err() {
        let (tmp, cwd, _global) = tree();
        let missing_global = tmp.path().join("absent");
        // Missing files and dirs are tolerated.
        let got = load_instructions(&cwd, &missing_global).unwrap();
        assert!(got.is_empty());
    }

    #[test]
    fn skills_frontmatter_and_shadowing() {
        let (tmp, cwd, global) = tree();
        let root = tmp.path();
        // Global skill.
        let gskill = global.join("skills/review");
        std::fs::create_dir_all(&gskill).unwrap();
        std::fs::write(
            gskill.join("SKILL.md"),
            "---\nname: review\ndescription: global reviewer\n---\nbody\n",
        )
        .unwrap();
        // Project skill shadowing the same name, discovered via .juto.
        let pskill = root.join("project/.juto/skills/review");
        std::fs::create_dir_all(&pskill).unwrap();
        std::fs::write(
            pskill.join("SKILL.md"),
            "---\nname: review\ndescription: project reviewer\n---\nbody\n",
        )
        .unwrap();
        // Second project skill, no frontmatter name → dir name.
        let other = cwd.join(".juto/skills/standup");
        std::fs::create_dir_all(&other).unwrap();
        std::fs::write(other.join("SKILL.md"), "---\ndescription: notes\n---\n").unwrap();

        let skills = discover_skills(&cwd, &global).unwrap();
        let by_name: BTreeMap<_, _> = skills.iter().map(|s| (s.name.as_str(), s)).collect();
        assert_eq!(by_name["review"].description, "project reviewer");
        assert!(by_name["review"].path.starts_with(root.join("project")));
        assert_eq!(by_name["standup"].name, "standup");
        assert_eq!(by_name["standup"].description, "notes");
    }

    #[test]
    fn malformed_frontmatter_errors() {
        let (tmp, cwd, global) = tree();
        let bad = cwd.join(".juto/skills/broken");
        std::fs::create_dir_all(&bad).unwrap();
        std::fs::write(bad.join("SKILL.md"), "---\nname: [unclosed\n---\n").unwrap();
        assert!(matches!(
            discover_skills(&cwd, &global),
            Err(ContextError::Frontmatter { .. })
        ));
        // No frontmatter at all → skipped, not an error.
        std::fs::write(bad.join("SKILL.md"), "plain markdown\n").unwrap();
        assert!(discover_skills(&cwd, &global).unwrap().is_empty());
        let _ = tmp;
    }

    #[test]
    fn runtime_installs_bundled_skill_before_first_turn() {
        let (tmp, cwd, global) = tree();
        crate::Runtime::from_config(&cwd, &global, crate::RuntimeConfig::default()).unwrap();

        let skill = discover_skills(&cwd, &global)
            .unwrap()
            .into_iter()
            .find(|skill| skill.name == "juto-repo-setup")
            .unwrap();
        assert!(skill.path.starts_with(global.join("bundled-skills")));
        assert!(
            std::fs::read_to_string(skill.path)
                .unwrap()
                .contains("## Procedure")
        );
        let _ = tmp;
    }
}
