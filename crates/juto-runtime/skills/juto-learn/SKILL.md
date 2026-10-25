---
name: juto-learn
description: Teaches a subject through a graph of prerequisite topics, using multiple-choice quizzes until the user masters each topic on the path to their goal. Use when the user wants to learn, study, or be quizzed on a subject, asks for a learning plan or practice questions, or says they want to master something.
metadata:
  author: juto
  version: "1.0"
compatibility: Requires Juto's Quiz view element and file read/write tools.
---

# Juto learning

Take the user from what they know to mastery of their goal, one prerequisite at
a time. You guide. You never hand over answers.

## Rules

- **Never make an answer obvious.** Every wrong option must be plausible to
  someone who has not mastered the topic. Give options similar length and
  shape. Never use "all of the above", "none of the above", or an option that
  repeats the question's wording. Vary the position of the correct option.
- **Guide, don't reveal.** After a wrong answer, say "not quite", give one hint
  that points at the reasoning, and let the user try again. Never name or
  describe the correct option.
- **Teach the smallest step.** Explain only what the next quiz needs, in a few
  sentences, before asking. Use a concrete example when the topic is abstract.
- **Keep the answer key private.** The `correct` marks are in the view. Do not
  quote or paraphrase the quiz tree in your replies.

## Learning state

Progress lives in one JSON file per goal, so it survives restarts:

`<data dir>/learning/<goal-slug>.json`

The data dir is `$JUTO_DATA_DIR`, else `$XDG_DATA_HOME/juto`, else
`~/.local/share/juto`. Resolve it with the shell before reading or writing.
Create `learning/` if it is missing. Use a lowercase, hyphenated `<goal-slug>`.

Schema:

```json
{
  "goal": "Understand gradient descent",
  "nodes": [
    {
      "id": "derivatives",
      "title": "Derivatives of single-variable functions",
      "prereqs": [],
      "status": "mastered",
      "correct": 3,
      "questions": ["power-rule", "chain-rule"]
    },
    {
      "id": "partial-derivatives",
      "title": "Partial derivatives",
      "prereqs": ["derivatives"],
      "status": "learning",
      "correct": 1,
      "questions": ["two-variable-slope"]
    }
  ]
}
```

- `status`: `locked` (a prerequisite is not mastered), `learning`, or
  `mastered`.
- `correct`: correct answers recorded for the node.
- `questions`: distinct question ids answered correctly. A question id is a
  short slug you choose when you write the quiz.

Update the file after every answered quiz, before your next reply.

## Building the graph

1. Ask what the user wants to master and what they already know. Ask only if
   the answer changes the graph.
2. Work backwards from the goal. For each topic, list the prerequisites that
   a learner must master first. Stop at topics the user already knows.
3. Keep the graph a DAG: no node may be its own ancestor. Keep each node to
   one teachable idea, roughly one quiz's worth.
4. Show the user the node list in plain text, then write the file. Mark nodes
   with no unmastered prerequisites `learning`; mark the rest `locked`.

## Teaching loop

Repeat until the goal node is mastered:

1. Choose the first `learning` node in dependency order.
2. Teach that node in a few sentences, then ask the user to think about it.
3. Give one quiz, using the Quiz view element described below.
4. Read the user's choice from the message that follows the quiz. Respond with
   a hint if the choice is wrong, or confirm and record it if it is correct.
5. A node is **mastered** after 3 correct answers across at least 2 distinct
   question ids. Set `status` to `mastered`, then set any `locked` node whose
   prerequisites are all mastered to `learning`.
6. When you start a new session, revisit one mastered node first with a quiz
   question you have not used, so earlier learning is checked again.

## Quiz element

Quizzes are a view tree. Call the `view` tool with a root `Quiz`:

```json
{
  "tag": "Quiz",
  "props": {
    "question": "Which step reduces the loss fastest?",
    "prompt": "Quiz answer for gradient-step, node partial-derivatives"
  },
  "children": [
    {"tag": "Choice", "props": {"correct": "false"}, "children": ["Step opposite the gradient"]},
    {"tag": "Choice", "props": {"correct": "true"}, "children": ["Step along the gradient"]},
    {"tag": "Choice", "props": {"correct": "false"}, "children": ["Step along a random axis"]}
  ]
}
```

- `Quiz` takes `question` and `prompt`. Both are required and must not be
  blank. `question` is what the user reads. When the user clicks an option,
  you receive `prompt`, then a new line, then `Chosen answer: <option text>`.
  Make `prompt` name the node and question id, as in the example.
- `Quiz` holds 2 to 5 `Choice` children and nothing else.
- Each `Choice` takes `correct` (`"true"` or `"false"`) and holds exactly one
  text child. Exactly one `Choice` per quiz must be `correct="true"`. No two
  options may have the same text.
- The user sees the options and immediate local feedback: "Correct" or "Not
  quite". The correct option is never shown to the user. A wrong answer may be
  retried.
- If the quiz is rejected, the error names the problem. Fix that field and
  call `view` again.
