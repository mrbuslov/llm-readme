# Content
- [Zero-/Few-shot Learning](#zero-few-shot-learning)
- [Chain-of-Thought Prompting](#chain-of-thought-prompting)
- [Agentic Prompting](#agentic-prompting)
- [Prompt Injection & Guardrails](#prompt-injection--guardrails)
- [Toxicity Filtering](#toxicity-filtering)
- [Hallucination Detection](#hallucination-detection)
- [Observability Tools (Langfuse, LangSmith, PromptLayer)](#observability-tools-langfuse-langsmith-promptlayer)
- [Integration SDKs (LangChain, Semantic Kernel, Haystack)](#integration-sdks-langchain-semantic-kernel-haystack)
- [RAG & System Prompt Fine-tuning](#rag--system-prompt-fine-tuning)
- [AI Governance & Risk Management](#ai-governance--risk-management)

---

# Zero-/Few-shot Prompting
This is the basics. The idea is simple: you give the model a few examples right in the prompt so it understands the pattern.

## Zero-shot Prompting
### The idea
The model solves a task with zero examples.  
You just tell it what to do, and it does it:  
**Input**:
```
Translate to French: Hello, how are you?
```
**Output**:
```
Bonjour, comment allez-vous?
```

### When it works:
Modern LLMs like GPT-4.1, Claude models (which I love) already saw billions of examples during training
Task is similar enough to what the model knows
Instructions are clear


## Few-shot Prompting 
### The idea
You give the model a few examples right in the prompt so it understands the pattern.  
**Input**:
```
Classify the sentiment of the review:

Review: "Great product, highly recommend!"
Sentiment: Positive

Review: "Terrible quality, returning it"
Sentiment: Negative

Review: "It's okay, could be better"
Sentiment: Neutral

Review: "Amazing! Best purchase ever!"
Sentiment:
```
**Output**:
```
Positive
```

### Number of examples:

One-shot: 1 example  
Few-shot: typically 2-10 examples  
Many-shot: 10+ examples (less common term)  

Why? :)
- You don't need to fine-tune the model
- No need for thousands of labeled examples
- Just change the prompt. Adjust behavior instantly
- Test different approaches in seconds
- Deploy immediately

## When to use what?
Use **zero-shot** when:
- Task is simple and common (translation, summarization)
- Model is powerful enough (GPT-4.1, Claude 3+)
- You want to save tokens/money

Use **few-shot** when:
- Task needs specific formatting
- Model struggles with zero-shot
- You have a unique pattern or style

---

# Chain-of-Thought Prompting
This is about making models think **STEP-BY-STEP**. The idea: instead of jumping straight to the answer, make the model show its **reasoning process**.

## What is Chain-of-Thought (CoT)?
### The idea
You ask the model to break down complex problems into intermediate steps before giving the final answer.  
Without CoT, models often fail at multi-step reasoning. With CoT, they "think out loud".

**Without CoT - Input**:
```
Roger has 5 tennis balls. He buys 2 more cans of tennis balls. 
Each can has 3 tennis balls. How many tennis balls does he have now?
```
**Without CoT - Output**:
```
11 tennis balls
```
(Model might get it wrong or right by luck)

**With CoT - Input**:
```
Roger has 5 tennis balls. He buys 2 more cans of tennis balls. 
Each can has 3 tennis balls. How many tennis balls does he have now?

Let's think step by step:
```
**With CoT - Output**:
```
Let's think step by step:
1. Roger starts with 5 tennis balls
2. He buys 2 cans of tennis balls
3. Each can has 3 balls, so 2 cans = 2 × 3 = 6 balls
4. Total = 5 + 6 = 11 tennis balls

Answer: 11 tennis balls
```

## Two types of CoT

### Zero-shot CoT
Just add "Let's think step by step" or "Let's break this down" to your prompt.  
**Input**:
```
What is 15% of 240? Let's think step by step.
```
**Output**:
```
Step 1: Convert percentage to decimal: 15% = 0.15
Step 2: Multiply: 240 × 0.15 = 36
Answer: 36
```

### Few-shot CoT
Show examples with step-by-step reasoning.  
**Input**:
```
Q: A store has 23 apples. They sell 17. How many are left?
A: Let's think step by step.
- Started with: 23 apples
- Sold: 17 apples
- Remaining: 23 - 17 = 6 apples
Answer: 6 apples

Q: John has $50. He spends $12 on lunch and $8 on coffee. How much does he have left?
A:
```
**Output**:
```
Let's think step by step.
- Started with: $50
- Spent on lunch: $12
- Spent on coffee: $8
- Total spent: $12 + $8 = $20
- Remaining: $50 - $20 = $30
Answer: $30
```

## When to use CoT?

Use **Chain-of-Thought** when:
- Math problems (especially multi-step)
- Logic puzzles
- Planning tasks
- Debugging code
- Any task requiring multiple reasoning steps
- Model gives wrong answers with direct prompting

Don't use CoT when:
- Simple factual questions
- Tasks that don't need reasoning
- You want fast, short responses
- Token cost matters a lot

## Why it works?
Models are better at reasoning when they externalize intermediate steps. It's like showing your work in math class - helps catch errors and follow logic.

The model uses its own output as additional context for the next part of reasoning.

## Pro tips
**Trigger phrases that work**:
- "Let's think step by step"
- "Let's break this down"
- "First, let's analyze..."
- "Let's solve this systematically"

**Structure your CoT prompts**:
```
Problem: [state the problem clearly]

Let's think step by step:
1. [first step]
2. [second step]
...

Answer: [final answer]
```

---

# Agentic Prompting
(description)

---

# Prompt Injection & Guardrails
(description)

---

# Toxicity Filtering
(description)

---

# Hallucination Detection
(description)

---

# Observability Tools (Langfuse, LangSmith, PromptLayer)
(description)

---

# Integration SDKs (LangChain, Semantic Kernel, Haystack)
(description)

---

# RAG & System Prompt Fine-tuning
(description)

---

# AI Governance & Risk Management
(description)
