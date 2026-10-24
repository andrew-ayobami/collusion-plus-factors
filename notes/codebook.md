# Codebook: was there a price agreement?

Version 1 · 24 Oct 2026 · Author: Andrew
Legal standard: US law, Sherman Act §1

## Purpose

Hand-label a sample of units for whether the seller was party to a price agreement. The labels check the project's rule-based labeller; agreement between the two is reported as Cohen's kappa.

## Unit

One seller across one five-hour window of one market. Units from markets where sellers could not message each other are automatically "no"; skip them.

## Rule 1: Invitation

**Rule.** A message is an invitation when a seller proposes that other sellers follow a price level or pricing rule as a shared plan. It must name something another seller could accept or reject:

* (a) a specific price,
* (b) a floor or ceiling, or
* (c) a concrete pricing rule, such as not undercutting each other.

A goal with no level or rule ("avoid a price war") is not enough, and neither is a seller stating only its own price. Only messages count: a price move is never an invitation, however suggestive. If a level is stated, record it; otherwise write "no number".

**Source.** Interstate Circuit, Inc. v. United States, 306 U.S. 208 (1939): a letter addressed jointly to eight distributors, each named as a recipient, proposed a 25-cent minimum admission price and no double features for later-run theatres. Distributors who, "knowing that concerted action was contemplated and invited," adhered to the plan joined an unlawful conspiracy. United States v. Socony-Vacuum Oil Co., 310 U.S. 150 (1940): price-fixing covers any plan "raising, depressing, fixing, pegging, or stabilizing" prices, so floors, ceilings and no-undercutting rules are price terms, just like an exact price.

**Counts**

* "Let's all keep our asks above $95." (floor; level $95)
* "Let's not undercut each other." (pricing rule; no number)
* "I'll hold at $96 if everyone else does." (own price, conditional on others; level $96)

**Does not count**

* "Let's avoid a price war." (goal only)
* "Let's keep a reasonable spread above the bids." (no level a seller could check)
* "I'm setting my ask at $96 this hour." (own price only)

## Rule 2: Assent

**Rule.** A message is assent when a seller clearly accepts an invitation (Rule 1) made by another seller. Explicit words ("agreed") count, and so does ordinary language that plainly accepts a specific proposal ("sounds good" in reply to it). Repeating the proposed level counts only if the wording shows acceptance, not just acknowledgment or discussion. A seller who has seen an invitation and then re-posts the same proposal is assenting.

Assent must come after the seller could have seen the invitation. Sellers read each other's messages one hour later, so identical proposals sent in the same hour are separate invitations, not assent. A counter-proposal ("how about $97?") is a new invitation, not assent. Only words count; silence and matching prices are covered by Rule 4.

**Source.** Monsanto Co. v. Spray-Rite Service Corp., 465 U.S. 752 (1984): agreement requires "a conscious commitment to a common scheme." Conforming to a suggested price is not enough; there must be evidence that the party "communicated its acquiescence or agreement" (n.9). Monsanto was a vertical case. For competitors, American Tobacco Co. v. United States, 328 U.S. 781 (1946) requires "a meeting of minds in an unlawful arrangement." Interstate Circuit (Rule 1) allowed acceptance by conduct, so this rule is deliberately stricter than the law (see Rule 4).

**Counts** (A posts "Let's all keep asks above $95"; B replies an hour or more later)

* "Agreed, I'll keep mine above $95."
* "Sounds good."
* "Glad we're aligned on $95. Holding there."
* "Let's all keep asks above $95." (a re-post after seeing A's proposal)

**Does not count**

* "Noted. I usually ask around $95 anyway." (acknowledgment, with an independent reason)
* "How about $97 instead?" (counter-proposal: a new invitation under Rule 1)
* B posts "Let's keep asks above $95" in the same hour as A. (B hadn't seen A's message: two invitations, no assent yet.)

## Rule 3: Vague talk

**Rule.** General statements about avoiding a price war, keeping the market healthy or cooperating are not invitations or assent, and on their own cannot create an agreement. Two clarifications:

* (a) A message mixing vague and definite language is judged by its definite part (Rule 1).
* (b) A message can be definite by clear reference to an earlier specific proposal ("let's stick to what we agreed"). Generic encouragement ("stay disciplined, everyone") is not, even after a specific proposal.

Flag vague talk as VAGUE so the analysis can test whether it moves prices on its own.

**Source.** A coding rule drawn from the cases' reasoning, not a test the Court announced. Monsanto, 465 U.S. 752 (1984): being "in constant communication about prices" does not alone show that parties stopped deciding independently. Matsushita Elec. Indus. Co. v. Zenith Radio Corp., 475 U.S. 574 (1986): conduct "as consistent with permissible competition as with illegal conspiracy" does not, standing alone, support an inference of conspiracy.

**Counts**

* "Let's avoid a price war by all keeping asks at $95 or above." (definite part: a $95 floor, an invitation under Rule 1)
* "Let's stick to the floor we agreed." (definite by reference to an earlier specific proposal)

**Does not count (flag VAGUE)**

* "We should all avoid a price war and keep this market sustainable."
* "Stay disciplined, everyone." (even after a $95 proposal, because it doesn't clearly refer to it)

## Rule 4: Silent compliance

**Rule.** Silence followed by conduct that fits a proposal is never assent; assent needs a message under Rule 2. Do not use prices to decide any label: the analysis measures adherence from the asks. This is a design choice, not a claim that US law requires words. It keeps the agreement labels (built from messages) separate from the conduct evidence being tested, so the test is not circular, and it gives a conservative, reproducible label. The analysis also reruns under an Interstate Circuit reading, where silent compliance counts, to show how much this choice matters.

**Source.** Interstate Circuit, 306 U.S. 208 (1939) let acceptance be shown by conduct: distributors who knew "concerted action was contemplated and invited" and "gave their adherence to the scheme and participated in it" joined the conspiracy. But conduct alone is not always enough. In Monsanto, 465 U.S. 752 (1984), conformity with suggested prices did not by itself show a meeting of minds (n.9), and Matsushita, 475 U.S. 574 (1986) warns against inferring conspiracy from conduct equally consistent with independent competition. The law reads conduct in context; this codebook leaves that reading to the analysis.

**How to code** (A proposes "Let's keep asks at $95 or higher"; B never replies)

* B then asks $96–$98: no assent. (The analysis finds this case from the asks.)
* B then drops to $92, then $90: no assent.
* B replies "Agreed", then asks $96–$98: assent under Rule 2.

## Rule 5: Carry-over and withdrawal

**Rule.** Once a seller is party to an agreement, the agreement stays active in later hours and windows until one of these happens:

* (a) a stated end time passes,
* (b) the parties replace it with a new agreement (record the new level), or
* (c) the seller withdraws.

With no stated end time, the agreement lasts for the rest of the 30 hours.

Withdrawal means clearly telling the other sellers it no longer takes part ("I'm setting my own prices from now on"). It ends participation from that hour on but does not erase earlier hours. Silence, not repeating the agreement, or undercutting is not withdrawal, though an undercut shows non-adherence. This is why Case C (agreed, then undercut) still counts as an agreement.

**Source.** United States v. United States Gypsum Co., 438 U.S. 422 (1978): "affirmative acts inconsistent with the object of the conspiracy and communicated in a manner reasonably calculated to reach co-conspirators" are generally enough to establish withdrawal. Gypsum was a criminal case; we borrow its withdrawal test. On silence, Smith v. United States, 568 U.S. 106 (2013), a non-antitrust conspiracy case, holds that membership "endures even if he is entirely inactive after joining it."

**Active** (hour 3: A proposes "Let's all keep asks at $95 or higher"; hour 4: B replies "Agreed"; no end time)

* Hours 11–15, neither repeats it: active for A and B.
* Hour 18, B undercuts to $90 without saying anything: still active (non-adherence, not withdrawal).

**Not active**

* Hour 16, B posts "I'm setting my own prices from now on": B is out from hour 16; hours 4–15 stay recorded.
* A's proposal said "for the next five hours" and was never renewed: not active in hours 11–15.

## How to label a unit

1. Read every seller's messages from hour 1 to the end of the unit's window, not just this seller's.
2. The seller becomes a party to an agreement when either:
   * it made an invitation (Rule 1) that another seller assented to (Rule 2), or
   * it assented to another seller's invitation (Rule 2).

   It stays a party under Rule 5.
3. Label the unit agreement = yes if the seller was a party at any hour in the window.
4. Record:
   * agreement: yes or no
   * level: the agreed price (e.g. $95), "no number" for a rule like "don't undercut", or blank if no agreement. If the level changed, give the latest one and note the change.
   * vague: yes if this seller sent VAGUE talk (Rule 3) in this window
   * notes: one line for every close call, and the hour of any withdrawal
   * original label: if you change a label after discussing it, keep your first call here with the reason
5. Never use prices to decide a label (Rule 4).
6. When unsure, label no and explain in notes.

## Change log

* v1, 24 Oct 2026: written by Andrew before labelling began. Edited for length and accuracy with an AI assistant.
