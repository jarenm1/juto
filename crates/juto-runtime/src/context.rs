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
    #[error("malformed SKILL.md frontmatter in {path}: {reason}")]
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
    let yaml = &rest[..end];
    let fm: SkillFrontmatter =
        serde_yaml::from_str(yaml).map_err(|e| ContextError::Frontmatter {
            path: path.to_path_buf(),
            reason: e.to_string(),
        })?;
    Ok(Some((fm.name, fm.description)))
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

/// Discover skills from `.juto/skills/<name>/SKILL.md` in every directory
/// from the filesystem root down to `cwd`, then `<global_dir>/skills/` and
/// `<global_dir>/.omp/skills/`. Nearer-to-cwd project skills shadow
/// same-named outer/global skills; global never overrides a project skill.
pub fn discover_skills(cwd: &Path, global_dir: &Path) -> Result<Vec<Skill>> {
    // First writer wins, so scan highest precedence first: project skills
    // from cwd outward, then the global dirs.
    let mut skills = BTreeMap::new();
    for dir in cwd.ancestors() {
        scan_skills_root(&dir.join(".juto/skills"), &mut skills)?;
        scan_skills_root(&dir.join(".omp/skills"), &mut skills)?;
    }
    scan_skills_root(&global_dir.join("skills"), &mut skills)?;
    scan_skills_root(&global_dir.join(".omp/skills"), &mut skills)?;
    Ok(skills.into_values().collect())
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
}
