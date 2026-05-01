"""The blueprint server: one loaded run, served for reading.

A separate process from the console on purpose. It holds a 32M-infoset table
resident and reaches into `engine`, which is exactly what the console must never
do -- and because the console talks to it over HTTP rather than importing it,
the `web_reads_through_the_command_layer` contract stays true rather than gaining
an exemption.

Where this process runs is a deployment question, not a design one: it needs the
run's checkpoint and the card abstraction, both of which live on the share.
"""
