"""Reading a trained blueprint: naming a spot, and what the strategy is there.

Everything here is a pure function of a loaded blueprint plus a description of a
spot. No HTTP, no session, no mutation -- so a server is a transport over this
package rather than a place where the answers are computed.
"""
