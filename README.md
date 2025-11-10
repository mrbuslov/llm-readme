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
(description)

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
