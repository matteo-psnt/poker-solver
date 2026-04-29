"""The terminal surface: argv in, rendered text and an exit status out.

Only ``headless`` lives here. The subcommands themselves are one level up in
``src.interfaces.commands``, shared with ``src.interfaces.web`` -- this package
is just the half that is genuinely terminal-specific, which is turning a
``CommandError`` into a message on stderr and exit 1. No other surface wants
that behaviour, and imposing it is what ``raise SystemExit`` at sixteen call
sites used to do.
"""
