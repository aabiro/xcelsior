#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaunchMode {
    Normal,
    Minimized,
}

pub fn launch_mode_from_env() -> LaunchMode {
    launch_mode_from_args(std::env::args())
}

pub fn launch_mode_from_args<I, S>(args: I) -> LaunchMode
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    if args.into_iter().any(|arg| arg.as_ref() == "--minimized") {
        LaunchMode::Minimized
    } else {
        LaunchMode::Normal
    }
}

#[cfg(test)]
mod tests {
    use super::{launch_mode_from_args, LaunchMode};

    #[test]
    fn defaults_to_normal_launch_mode() {
        let args = vec!["xcelsior-desktop".to_string()];
        assert_eq!(launch_mode_from_args(args), LaunchMode::Normal);
    }

    #[test]
    fn detects_minimized_launch_mode() {
        let args = vec!["xcelsior-desktop".to_string(), "--minimized".to_string()];
        assert_eq!(launch_mode_from_args(args), LaunchMode::Minimized);
    }
}
