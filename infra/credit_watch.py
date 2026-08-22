#!/usr/bin/env python3
"""Watch the two things that decide whether the credit card is ever charged.

This subscription has NO hard spending cap and cannot be given one: the Azure
spending limit is unavailable for Azure Plan / pay-as-you-go pricing, which is
what an MCA subscription is. A MasterCard is attached to the billing profile.
So the card is reachable by exactly two routes, and this script watches both.

Route 1 -- the credit runs out or expires
    Charges land on the card once the $10k `Azure for startups credit` lot is
    exhausted or past its expiry. Reported here as WORST-CASE RUNWAY: the
    balance divided by max_nodes x per-node rate, i.e. how long a total,
    unnoticed runaway would take to reach the card. Not a forecast of actual
    spend -- a floor on how much warning you have.

Route 2 -- charges that were never credit-eligible
    Marketplace images, third-party services, support plans and Entra ID P1/P2
    bypass Azure credits entirely and bill the card TODAY, with $9.9k sitting
    unused. Nothing in the monthly budget can see this: a budget measures burn,
    not eligibility, and would sit quiet while such a charge went through.

Detection
---------
Route 2 is checked two ways, deliberately redundant:

  LEADING   cost grouped by PublisherType and ChargeType. Anything outside the
            known-good allowlist alerts. Allowlist rather than a match against
            'Marketplace' on purpose -- the enum is not fully documented, and an
            unfamiliar value should wake you up rather than pass silently.

  DEFINITIVE any invoice with amountDue > 0. Unambiguous, catches paths nobody
            enumerated, but only lands monthly. The leading check covers the gap.

An earlier design compared total cost against `pendingEligibleCharges` and
alerted on any gap. It is not used: that field covers all UNINVOICED charges
while a cost query covers a date range, so the two only agree until the first
invoice closes. Reconstructing invoice boundaries to keep them comparable is
fragile, and the grouping above answers the same question directly. The gap is
still printed, as information rather than as an alert.

Fail closed
-----------
Three exit states, not two. A watchdog that dies silently on an expired CLI
token looks exactly like an all-clear, which is worse than no watchdog at all
for the one question this exists to answer.

    0   clear
    1   alert -- a human needs to look
    3   COULD NOT EVALUATE -- auth, API error, or an unrecognised response shape

Any caller must treat 3 as a problem, not as success.

Deliberately does not act
-------------------------
No auto-`just panic`. Cost data lags hours and gets restated, and at ~130 days
of runway there is no scenario where acting on a stale number beats reading the
alert. `just panic` stays the deliberate manual lever it already is.

Not a substitute for the real controls
--------------------------------------
This runs wherever you run it, which means it misses exactly the unattended
weekend it exists for. What protects you regardless is `max_nodes` bounding the
burn RATE, the Azure-side budget emails, and the pool holding zero nodes at
rest. This is additive.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

SUBSCRIPTION_ID = "f9c31345-15ac-413f-8841-5d0151baca66"

ALLOWED_PUBLISHER_TYPES = {"microsoft", "azure"}
# `roundingadjustment` judged and admitted 2026-08-09, which is what the
# allowlist is FOR: an unfamiliar value woke us up, someone looked, and it goes
# in. It appeared as -$0.07 the day the first invoice closed -- Azure's own
# reconciliation of sub-cent usage against the invoiced total. Left out, it
# alerted on every run forever, and a watchdog that always fires is one nobody
# reads. Note it is also NEGATIVE, which no charge to a card can be.
ALLOWED_CHARGE_TYPES = {"usage", "unusedreservation", "roundingadjustment", ""}

BILLING_API = "2024-04-01"
CONSUMPTION_API = "2023-05-01"
COSTMGMT_API = "2023-03-01"

EXIT_CLEAR = 0
EXIT_ALERT = 1
EXIT_UNEVALUABLE = 3

EPOCH_FALLBACK = "2026-07-01"


class UnevaluableError(Exception):
    """Raised when a check cannot be run at all, as opposed to running and passing."""


@dataclass
class Report:
    """Accumulated findings. `alerts` decides the exit code; `lines` is the output."""

    lines: list[str] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)

    def say(self, text: str = "") -> None:
        self.lines.append(text)

    def alert(self, text: str) -> None:
        self.alerts.append(text)
        self.lines.append(f"  ALERT  {text}")

    def absorb(self, other: Report) -> None:
        """Take another report's findings. How a concurrent check reports back."""
        self.lines.extend(other.lines)
        self.alerts.extend(other.alerts)


def concurrently(thunks: list[Callable[[], object]]) -> None:
    """Run independent checks at once, and fail as though they had not been.

    Each check is its own `az` process and its own round trip, and none needs
    another's answer -- 14.8s of 17.4s was spent waiting. Nothing here is CPU-bound.

    Two properties are preserved deliberately, because this is a watchdog:

    order
        Each check writes into its OWN report and the caller merges them, so output
        is byte-identical to the sequential version. A report that reorders itself
        run to run cannot be diffed, which is most of what a person does with one.
    the first failure, by DECLARATION
        Not the first raised. The same broken environment must produce the same
        message every time, and which of five concurrent calls loses its token
        first is a race.
    """
    with ThreadPoolExecutor(max_workers=len(thunks) or 1) as pool:
        futures = [pool.submit(thunk) for thunk in thunks]
        errors = [_exception_of(future) for future in futures]
    for error in errors:
        if error is not None:
            raise error


def _exception_of(future: Future[object]) -> BaseException | None:
    """Wait for a check and hand back what it raised, if anything.

    Swallowed here rather than propagated so that every check is WAITED for
    before any failure is reported: raising from inside the loop would leave
    the rest running against an `az` process nobody reads.
    """
    try:
        future.result()
    except Exception as exc:  # noqa: BLE001 -- re-raised in declaration order by the caller
        return exc
    return None


"""Being told to slow down is not the same as being unable to answer
------------------------------------------------------------------
Cost Management rate-limits its query API hard and per-tenant, and answers 429
rather than queueing. Observed: running this three times inside half a minute
was enough. Reported as COULD NOT EVALUATE that reads exactly like an expired
token or an unreachable Azure -- a watchdog crying wolf at the one thing it is
supposed to be trusted about.

Backed off and retried instead, because 429 is the one failure here that is
KNOWN to be temporary. Everything else still fails immediately: a wrong URL or
a lost token will not improve by being asked again, and a watchdog that takes a
minute to report a real problem is its own kind of broken.
"""
_THROTTLE_BACKOFF_SECONDS = (2.0, 6.0, 15.0)


def _is_throttled(proc: subprocess.CompletedProcess[str]) -> bool:
    """Whether Azure refused because it is being asked too often."""
    detail = f"{proc.stderr} {proc.stdout}".lower()
    return "too many requests" in detail or '"429"' in detail


def az_json(args: list[str], what: str) -> dict:
    """Run an `az` command expected to emit JSON, or raise UnevaluableError."""
    for pause in (*_THROTTLE_BACKOFF_SECONDS, None):
        try:
            proc = subprocess.run(
                ["az", *args, "-o", "json"],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except FileNotFoundError as exc:
            raise UnevaluableError("the `az` CLI is not installed or not on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise UnevaluableError(f"{what}: timed out after 120s") from exc

        if proc.returncode == 0:
            break
        if _is_throttled(proc):
            if pause is not None:
                time.sleep(pause)
                continue
            # Named as throttling rather than reported as a raw error, because
            # the two call for opposite responses: this one is "wait and run it
            # again", and every other COULD-NOT-EVALUATE is "something is
            # broken". Still exit 3 -- an unanswered question is unanswered.
            raise UnevaluableError(
                f"{what}: Azure is rate-limiting this query and did not relent after "
                f"{sum(_THROTTLE_BACKOFF_SECONDS):.0f}s of backoff. Nothing is wrong with "
                "the credential or the account — wait a minute and run it again."
            )
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else f"exit {proc.returncode}"
        raise UnevaluableError(f"{what}: {tail}")

    if not proc.stdout.strip():
        return {}

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise UnevaluableError(f"{what}: response was not JSON") from exc


def az_get(url: str, what: str) -> dict:
    return az_json(["rest", "--method", "get", "--url", url], what)


def check_auth() -> str:
    """Prove the CLI is logged in BEFORE any query, so a token failure is not
    mistaken for a clean result. Returns the signed-in identity, which is
    printed: this tenant holds two subscriptions, and reading a clean report
    against the wrong one is the failure this makes visible."""
    account = az_json(["account", "show"], "az account show")
    if not account.get("id"):
        raise UnevaluableError("not logged in — run `az login`")
    return account.get("user", {}).get("name") or account["id"]


@dataclass(frozen=True)
class Profile:
    """Where this subscription is billed, and since when.

    `created` dates the billing account, which is the natural start for the
    charge-shape scan: no charge can predate it, so scanning from there covers
    every billing period without hardcoding a date that silently goes stale.
    """

    account: str
    profile: str
    agreement: str
    created: str


def discover_profile(subscription_id: str) -> Profile:
    """Find the billing account and the billing profile that bills this
    subscription. Discovered rather than hardcoded so the script keeps working
    if the credit grant moves to a different profile."""
    accounts = az_json(["billing", "account", "list"], "billing account list")
    rows = accounts if isinstance(accounts, list) else accounts.get("value", [])
    if not rows:
        raise UnevaluableError("no billing account visible to this identity")

    # Asked of every account at once, then scanned in the listing's order. One
    # account is the normal case and this changes nothing; several made the
    # search cost a round trip per account before it could even start looking,
    # and which account bills this subscription is not knowable in advance.
    listings: list[list[dict]] = [[] for _ in rows]

    def fetch(index: int, ba: str) -> Callable[[], object]:
        def _load() -> None:
            url = (
                f"https://management.azure.com/providers/Microsoft.Billing"
                f"/billingAccounts/{ba}/billingSubscriptions?api-version={BILLING_API}"
            )
            listings[index] = az_get(url, "billingSubscriptions").get("value", [])

        return _load

    concurrently([fetch(index, account["name"]) for index, account in enumerate(rows)])

    for account, subs in zip(rows, listings, strict=True):
        ba = account["name"]
        for sub in subs:
            props = sub.get("properties", {})
            if props.get("subscriptionId") != subscription_id:
                continue
            profile_id = props.get("billingProfileId", "")
            if not profile_id:
                raise UnevaluableError(f"subscription {subscription_id} has no billing profile")
            created = (account.get("systemData", {}).get("createdAt") or "")[:10]
            return Profile(
                account=ba,
                profile=profile_id.rsplit("/", 1)[-1],
                agreement=account.get("agreementType", "unknown"),
                created=created or EPOCH_FALLBACK,
            )

    raise UnevaluableError(
        f"subscription {subscription_id} is not billed by any visible billing account"
    )


def check_credit(report: Report, ba: str, bp: str, daily_burn: float, warn_days: int) -> None:
    """Route 1: how much credit is left, and how long it would survive a total
    runaway. Uses estimatedBalance, NOT currentBalance -- the latter only moves
    when an invoice closes and reads as the full grant until then."""
    url = (
        f"https://management.azure.com/providers/Microsoft.Billing"
        f"/billingAccounts/{ba}/billingProfiles/{bp}"
        f"/providers/Microsoft.Consumption/credits/balanceSummary"
        f"?api-version={CONSUMPTION_API}"
    )
    props = az_get(url, "credits balanceSummary").get("properties", {})
    summary = props.get("balanceSummary", {})

    estimated = summary.get("estimatedBalance", {}).get("value")
    current = summary.get("currentBalance", {}).get("value")
    pending = props.get("pendingEligibleCharges", {}).get("value", 0.0)

    if estimated is None:
        raise UnevaluableError("balanceSummary has no estimatedBalance — response shape changed")

    runway = estimated / daily_burn if daily_burn > 0 else float("inf")

    report.say(f"  credit remaining   ${estimated:,.2f}   (last closed: ${current or 0:,.2f})")
    report.say(f"  uninvoiced charges ${abs(pending):,.2f}")
    report.say(f"  worst-case runway  {runway:,.0f} days at ${daily_burn:,.2f}/day")

    if runway < warn_days:
        report.alert(
            f"under {warn_days} days of worst-case runway "
            f"(${estimated:,.2f} left, ${daily_burn:,.2f}/day)"
        )


def check_expiry(report: Report, ba: str, bp: str, warn_days: int) -> None:
    """Route 1, second failure mode. Expiry and depletion are different events
    and get separate triggers. Earliest expiry across all lots, so a second
    grant landing later cannot mask a nearer deadline."""
    url = (
        f"https://management.azure.com/providers/Microsoft.Billing"
        f"/billingAccounts/{ba}/billingProfiles/{bp}"
        f"/providers/Microsoft.Consumption/lots?api-version={CONSUMPTION_API}"
    )
    lots = az_get(url, "consumption lots").get("value", [])
    if not lots:
        report.alert("no credit lots found — every charge now goes to the payment method")
        return

    now = datetime.now(UTC)
    soonest: tuple[datetime, str] | None = None

    for lot in lots:
        props = lot.get("properties", {})
        raw = props.get("expirationDate")
        if not raw:
            continue
        # `fromisoformat` parses the trailing `Z` natively on 3.11+, with or
        # without fractional seconds. The hand-rolled
        # `.replace("Z","+00:00").split(".")[0] + "+00:00"` only worked when a
        # fraction was present; on the plain `...T00:00:00Z` Azure usually
        # returns it produced a DOUBLE offset and raised. The broad
        # `except Exception` in main() then turned that into exit 3, so the
        # credit-expiry half of the watchdog stopped answering silently.
        stamp = datetime.fromisoformat(raw)
        source = props.get("source", lot.get("name", "?"))
        if soonest is None or stamp < soonest[0]:
            soonest = (stamp, source)

    if soonest is None:
        report.say("  credit expiry      none recorded")
        return

    stamp, source = soonest
    days = (stamp - now) / timedelta(days=1)
    report.say(f"  credit expires     {stamp:%Y-%m-%d}  ({days:,.0f} days)  [{source}]")

    if days < warn_days:
        report.alert(f"credit expires in {days:,.0f} days ({stamp:%Y-%m-%d})")


def check_charge_shape(report: Report, ba: str, bp: str, since: str) -> None:
    """Route 2, leading indicator. Anything outside the allowlist is a charge
    that may not be credit-eligible, which means it reaches the card while the
    balance sits untouched."""
    url = (
        f"https://management.azure.com/providers/Microsoft.Billing"
        f"/billingAccounts/{ba}/billingProfiles/{bp}"
        f"/providers/Microsoft.CostManagement/query?api-version={COSTMGMT_API}"
    )
    body = {
        "type": "ActualCost",
        "timeframe": "Custom",
        "timePeriod": {
            "from": f"{since}T00:00:00Z",
            "to": f"{datetime.now(UTC):%Y-%m-%d}T23:59:59Z",
        },
        "dataset": {
            "granularity": "None",
            "aggregation": {"totalCost": {"name": "Cost", "function": "Sum"}},
            "grouping": [
                {"type": "Dimension", "name": "PublisherType"},
                {"type": "Dimension", "name": "ChargeType"},
            ],
        },
    }
    result = az_json(
        ["rest", "--method", "post", "--url", url, "--body", json.dumps(body)],
        "cost query",
    )
    props = result.get("properties", {})
    columns = [c["name"] for c in props.get("columns", [])]
    rows = props.get("rows", [])

    for needed in ("Cost", "PublisherType", "ChargeType"):
        if needed not in columns:
            raise UnevaluableError(
                f"cost query returned no {needed} column — response shape changed"
            )

    idx = {name: i for i, name in enumerate(columns)}
    total = 0.0

    report.say(f"  charges since {since}:")
    for row in rows:
        cost = float(row[idx["Cost"]])
        publisher = str(row[idx["PublisherType"]] or "")
        charge = str(row[idx["ChargeType"]] or "")
        total += cost

        ok = publisher.lower() in ALLOWED_PUBLISHER_TYPES and charge.lower() in ALLOWED_CHARGE_TYPES
        mark = "   " if ok else ">> "
        report.say(f"    {mark}${cost:>10,.2f}  {publisher or '(none)'} / {charge or '(none)'}")

        if not ok:
            report.alert(
                f"${cost:,.2f} of {publisher or '(none)'}/{charge or '(none)'} charges — "
                f"outside the credit-eligible allowlist, so this may bill the card directly"
            )

    report.say(f"       ${total:>10,.2f}  total")


def check_invoices(report: Report, ba: str) -> None:
    """Route 2, definitive. amountDue > 0 means real money is owed on the
    payment method. Monthly latency, which is why the leading check exists."""
    url = (
        f"https://management.azure.com/providers/Microsoft.Billing"
        f"/billingAccounts/{ba}/invoices?api-version={BILLING_API}"
        f"&periodStartDate=2020-01-01&periodEndDate={datetime.now(UTC):%Y-%m-%d}"
    )
    invoices = az_get(url, "invoices").get("value", [])

    if not invoices:
        report.say("  invoices           none issued yet")
        return

    for invoice in invoices:
        props = invoice.get("properties", {})
        due = (props.get("amountDue") or {}).get("value", 0.0) or 0.0
        end = (props.get("billingPeriodEndDate") or "?")[:10]
        status = props.get("status", "?")
        report.say(f"  invoice {end}  status={status}  amountDue=${due:,.2f}")
        if due > 0:
            report.alert(
                f"invoice period ending {end} has ${due:,.2f} due — "
                f"this is charged to the payment method, not the credit"
            )


def check_payment_methods(report: Report, ba: str, bp: str) -> None:
    """Informational. Not an alert -- a card being present is the expected
    state, and there is no API to remove it. Printed so a card appearing,
    changing, or vanishing is visible rather than discovered on an invoice."""
    url = (
        f"https://management.azure.com/providers/Microsoft.Billing"
        f"/billingAccounts/{ba}/billingProfiles/{bp}"
        f"/paymentMethodLinks?api-version={BILLING_API}"
    )
    links = az_get(url, "paymentMethodLinks").get("value", [])
    if not links:
        report.say("  payment method     NONE attached")
        return
    for link in links:
        props = link.get("properties", {})
        report.say(
            f"  payment method     {props.get('displayName', '?')} "
            f"...{props.get('lastFourDigits', '????')} "
            f"exp {props.get('expiration', '?')} ({props.get('status', '?')})"
        )


def unevaluable(reason: str) -> int:
    print("CREDIT WATCH — COULD NOT EVALUATE", file=sys.stderr)
    print(f"  {reason}", file=sys.stderr)
    print("  Treat this as a FAILURE, not an all-clear.", file=sys.stderr)
    return EXIT_UNEVALUABLE


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Alert before Azure charges can reach the credit card.",
    )
    parser.add_argument(
        "--daily-burn",
        type=float,
        default=576.00,
        help="worst-case $/day: max_nodes x per-node rate. Raise this whenever "
        "max_nodes or pool_vm_size goes up.",
    )
    # Priced at the $0.80/hr/node LIST rate, not the $0.688 measured against
    # Cost Management: this figure divides the credit balance to produce runway,
    # so over-stating the burn under-states the warning time, which is the safe
    # direction. It is the one line here not meant to be exact -- a runaway is
    # not obliged to stop at max_nodes-worth of compute, and storage and egress
    # ride along on top.
    #
    # 76.80 at max_nodes=4; 307.20 at 16 (2026-08-15); 576.00 at 30 (2026-08-22,
    # with the quota raise 256 -> 512). THIS IS NOT DERIVED FROM TERRAFORM and
    # nothing fails if the two drift -- the runway number simply becomes
    # optimistic, which is the one direction that matters. At this burn a ~$9.0k
    # balance is ~16 days, so `--warn-days 30` now fires: the ceiling was chosen
    # deliberately and the alert is meant to ask whether the credit is being
    # spent on purpose.
    parser.add_argument(
        "--warn-days",
        type=int,
        default=30,
        help="alert below this many days of worst-case runway (default 30)",
    )
    parser.add_argument(
        "--expiry-warn-days",
        type=int,
        default=60,
        help="alert when the nearest credit lot expires within this many days",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="start date for the charge-shape scan (YYYY-MM-DD). Defaults to the "
        "billing account's creation date, so the scan always covers every billing "
        "period rather than a hardcoded date that goes stale.",
    )
    parser.add_argument(
        "--subscription",
        default=SUBSCRIPTION_ID,
        help="subscription whose billing profile is watched",
    )
    args = parser.parse_args()

    report = Report()
    try:
        identity = check_auth()
        found = discover_profile(args.subscription)
        since = args.since or found.created
        report.say(f"  signed in as       {identity}")
        report.say(f"  billing profile    {found.profile} ({found.agreement})")
        report.say()

        # Five independent questions, asked at once and printed in this order.
        # `check_auth` above stays sequential and first on purpose: it is the
        # fail-closed gate, and it also settles the CLI's token before five
        # processes want it at the same moment.
        payment, credit, expiry, invoices, charges = (Report() for _ in range(5))
        concurrently(
            [
                lambda: check_payment_methods(payment, found.account, found.profile),
                lambda: check_credit(
                    credit, found.account, found.profile, args.daily_burn, args.warn_days
                ),
                lambda: check_expiry(expiry, found.account, found.profile, args.expiry_warn_days),
                lambda: check_invoices(invoices, found.account),
                lambda: check_charge_shape(charges, found.account, found.profile, since),
            ]
        )
        for section in (payment, credit, expiry):
            report.absorb(section)
        report.say()
        report.absorb(invoices)
        report.say()
        report.absorb(charges)
    except UnevaluableError as exc:
        return unevaluable(str(exc))
    except Exception as exc:  # noqa: BLE001 -- a crash must not read as a real alert; see below
        # Anything unexpected is COULD-NOT-EVALUATE, not ALERT. Letting it
        # propagate would exit 1, which this script defines as "alert" -- so a
        # crash would be indistinguishable from a real finding, and a caller
        # keying on exit codes would report a card breach because of a typo.
        return unevaluable(f"unexpected {type(exc).__name__}: {exc}")

    print("CREDIT WATCH")
    for line in report.lines:
        print(line)
    print()

    if report.alerts:
        print(f"  {len(report.alerts)} ALERT(S) — the card is reachable or will be:")
        for alert in report.alerts:
            print(f"    - {alert}")
        print()
        print("  To stop all spend now:  just panic")
        return EXIT_ALERT

    print("  clear — all charges credit-eligible, nothing due on the card")
    return EXIT_CLEAR


if __name__ == "__main__":
    sys.exit(main())
