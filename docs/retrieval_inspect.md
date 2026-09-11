# Retrieval inspection — 10 production-path cases (intent-filtered top-3)

Method: `retrieve(client, text, intent_filter=true_intent, top_k=3)` — the live
path. Honest notes, including where retrieval teaches bad habits.

1. **GS_0042** (refund gibberish → sim 0.8033): top-1 is another "not refunding
   my money" case with a DM-redirect reply. Relevant: same ask, same move.
   Draft mirrors it safely.
2. **GS_0121** (Fire Stick network → 0.7876): top-1 is a Fire TV Stick
   replace-question. Relevant device family. Note: historical reply contains a
   `t.co` link — the corpus models link-sharing our guardrail forbids.
3. **GS_0012** (app cancel white-screen → 0.7570): top-1 is the same broken
   cancel-button issue. Explains the old hallucination: the drafter COPIED the
   `t.co` chat link from this retrieved reply. Guardrail strips it anyway.
4. **GS_0001** (repeat delivery delays → 0.7934): top-1 is a multi-delay
   complaint. Relevant pattern, wrong specifics — draft correctly stays generic.
5. **GS_0132** (Echo profanity quirk → 0.7142): top-1 is an Alexa language
   complaint ("whiting up my language"). Semantically nearest available, but
   the issues differ — low-sim retrieval (0.71) still grounds tone, not facts.
6. **GS_0159** (wrong hot sauce → 0.7732): top-1 is a wrong-item replacement
   case. Directly relevant; draft follows its DM-redirect shape.
7. **GS_0093** (Pay balance deducted → 0.7198): top-1 is another deducted-amount
   case. Relevant; historical reply again link-heavy.
8. **GS_0233** (German app status bug → 0.7708): top-1 is a German delivery-date
   complaint. Cross-lingual matching works — same language, adjacent topic.
9. **GS_0160** (birthday gift failure → 0.7972): top-1 is a delay complaint with
   an empathetic reply. Tone-relevant; draft adds case-specific empathy the
   retrieval only sketches.
10. **GS_0119** (Prime trial bank chaos → 0.7579): top-1 is a post-trial bank
    charge case (£7.99). Strong match on facts and stakes.

Aggregate context (`results/retrieval_eval.json`, unfiltered): consistency@1
0.49 / @3 0.685 / @5 0.755; mean top-5 similarity 0.78. The intent filter
exists because unfiltered top-1 is wrong half the time (DELIVERY_DAMAGE: 0.00).
Corpus caveat: historical replies routinely contain short-links and "send us
your info" asks — patterns our safety policy forbids the drafter from copying.
