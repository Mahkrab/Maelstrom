//! The binary adapts process arguments and signals to the headless library.

mod cli;

fn main() -> std::process::ExitCode {
    cli::main()
}
