"""
tools/test_chat.py
==================
Put the assistant through the questions people actually ask.

    python tools/test_chat.py              # routing only, no model needed
    python tools/test_chat.py --live       # also run the SQL fallback
    python tools/test_chat.py --ask "..."  # one question

What is being checked
---------------------
Mostly ROUTING. The value of the deterministic layer is that a question
reaches the handler that knows how to answer it, so the test asserts
which handler answered rather than checking the wording - wording is
allowed to improve, routing is not allowed to drift.

The cases below are the ones that were asked for, plus the ones that
tend to break a system like this: a name misspelled, a camera referred
to three different ways, a question about somebody who does not exist,
and a question deliberately outside what any handler covers so the
fallback is exercised.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (question, expected handler or None to mean "anything sensible")
CASES = [
    # ---- the ones that were asked for -------------------------------
    ("how many cams are running and what are they", "cameras"),
    ("who are present in reception_lobby cam now", "who_present"),
    ("whats the last one hour summary of the cam2", "camera_summary"),
    ("what is the time of Deepthi first entry", "first_entry"),
    ("what is the time of deepthi last entry", "last_entry"),
    ("how many members present in cam2 and who are they", None),

    # ---- cameras ----------------------------------------------------
    ("list all cameras", "cameras"),
    ("what cameras do we have", "cameras"),
    ("how many cameras are configured", "cameras"),

    # ---- who is here ------------------------------------------------
    ("who is in the building right now", "who_present"),
    ("who is present now", "who_present"),
    ("who is in server rm psg", "who_present"),
    ("is Akhila here right now", "is_here"),
    ("is Srikanth Nandigama still around", "is_here"),

    # ---- counting ---------------------------------------------------
    ("how many people are present now", "count_present"),
    ("how many people entered today", "entries_count"),

    # ---- one person -------------------------------------------------
    ("when did Akhila first arrive today", "first_entry"),
    ("what time did Ayesha last enter", "last_entry"),
    ("show all entries for Shivajyothi", "entries"),
    ("how long has Akhila been present today", "person_time"),

    # ---- a misspelling ----------------------------------------------
    ("what time did Akila first enter", "first_entry"),

    # ---- summaries and rankings -------------------------------------
    ("give me a summary of the last 2 hours", "camera_summary"),
    ("summary of reception today", "camera_summary"),
    ("who spent the most time today", "top_presence"),
    ("who has not come in today", "absent"),
    ("what happened in the last 30 minutes", "recent_activity"),

    # ---- housekeeping -----------------------------------------------
    ("how many people are enrolled", "employees"),
    ("how many confirmations are pending", "reviews"),
    ("is everything running ok", "status"),

    # ---- LOOKING at a live frame (core/vlm_scene.py) ----------------
    # These are not answered from the database, so the routing test
    # below never CALLS them - see LIVE_HANDLERS. What is checked here
    # is the thing that actually broke when they were added: whether a
    # question about what people are doing still reaches them, and
    # whether the older questions that share their words still reach the
    # database handlers above.
    ("whats happening in server_psg_cam", "scene"),
    ("what is going on in the reception right now", "scene"),
    ("describe server rm psg", "scene"),
    ("pavan ayeesha what are they doing", "person_activity"),
    ("what is Akhila doing", "person_activity"),
    ("is Ayesha Fatima working right now", "person_activity"),
    ("who is working and who is not working", "working"),
    ("who are working now", "working"),
    ("is anybody idle", "working"),

    # ...and the pairs these have to stay separable from. Each one uses
    # a word from a live handler's trigger list and belongs somewhere
    # else: "working" (system health), "happened" (the past), "doing"
    # with a time question in front of it.
    ("is everything working", "status"),
    ("what happened in the last hour", "recent_activity"),
    ("how long has Akhila been present today", "person_time"),
    ("what is the last one hour summary of cam2", "camera_summary"),

    # ---- nobody by that name ----------------------------------------
    ("what time did Nonexistent Person arrive", None),

    # ---- outside every handler: the SQL fallback --------------------
    ("which camera has recorded the most visits ever", None),
    ("list the five longest presence sessions this week", None),
]


# Handlers that LOOK at a live camera instead of reading the database.
#
# The routing test runs each handler to find out what it answers with,
# which is free for a database handler and emphatically not free for
# these: one of them is a running pipeline, a GPU and up to
# VLM_SCENE_BUDGET_SECONDS away from returning. So they are routed to
# and not called, unless --live says otherwise, and their route is taken
# from the handler's name.
#
# Which means the plain run stays what it has always been: a fast check
# that nothing needs to be up for.
LIVE_HANDLERS = {
    "answer_scene": "scene",
    "answer_person_activity": "person_activity",
    "answer_working": "working",
}


def main():
    parser = argparse.ArgumentParser(description="Exercise the assistant")
    parser.add_argument("--live", action="store_true",
                        help="also run questions that need the model")
    parser.add_argument("--ask", default=None, help="ask one question")
    parser.add_argument("--verbose", action="store_true",
                        help="print the full answer for every case")
    args = parser.parse_args()

    from core.chat import Assistant
    from core.chat_intents import match

    if args.ask:
        assistant = Assistant()
        answer = assistant.ask(args.ask)
        print(f"\nQ: {args.ask}")
        print(f"A: {answer.text}")
        if answer.rows:
            from core.chat import _table
            print()
            print(_table(answer.columns, answer.rows))
        if answer.sql:
            print(f"\nSQL: {answer.sql}")
        print(f"\n[via {answer.source}, {len(answer.rows)} row(s)]")
        return 0

    # Routing is checked WITHOUT the model, so this stays runnable when
    # Ollama is not up - the deterministic layer is the part that must
    # never regress.
    assistant = Assistant(allow_sql=args.live, phrase=args.live)

    passed = failed = fell_through = 0
    print(f"{'question':<52}{'handler':<18}{'result'}")
    print("-" * 88)
    for question, expected in CASES:
        handler, kwargs = match(question)
        route = None
        live_route = LIVE_HANDLERS.get(getattr(handler, "__name__", ""))
        if live_route and not args.live:
            answer = None
            route = live_route          # routed to, deliberately not run
        elif handler is not None:
            answer = handler(question, **kwargs)
            route = answer.source
        elif args.live:
            answer = assistant.ask(question)
            route = answer.source
        else:
            answer = None
            route = "(sql fallback)"

        if expected is None:
            verdict = "ok"
            if handler is None:
                fell_through += 1
            passed += 1
        elif route == expected:
            verdict = "ok"
            passed += 1
        else:
            verdict = f"EXPECTED {expected}"
            failed += 1

        print(f"{question[:51]:<52}{str(route)[:17]:<18}{verdict}")
        if args.verbose and answer is not None:
            print(f"      -> {answer.text[:160]}")

    print("-" * 88)
    print(f"{passed} passed, {failed} misrouted, "
          f"{fell_through} left to the SQL fallback")
    if not args.live:
        print("\n(routing only - add --live to exercise the model and the "
              "SQL fallback)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
