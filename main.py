import argparse

from crew.graph import build_graph, initial_state
from crew.llm import usage
from crew.report import save


def main():
    parser = argparse.ArgumentParser(description="Verdict: a research crew that checks its own facts")
    parser.add_argument("question", help="the question to research, in quotes")
    args = parser.parse_args()

    state = build_graph().invoke(initial_state(args.question))
    report = save(state, usage)

    print("\n" + "=" * 60 + "\n" + state["answer"] + "\n" + "=" * 60)
    print(f"Report: {report}")
    print(f"LLM calls: {usage['calls']}, tokens in/out: "
          f"{usage['input_tokens']}/{usage['output_tokens']}")


if __name__ == "__main__":
    main()
