# Arceus: A Meta-Cognitive Approach to ARC-AGI Problem Solving

**Technical Documentation**

## Abstract

Arceus is a sophisticated ARC-AGI puzzle solver that employs a three-layer meta-cognitive architecture combined with an external memory system to solve abstract reasoning tasks. Unlike traditional solvers that mechanically try strategies until one succeeds, Arceus reasons about how to approach problems, adapts its memory retrieval based on problem characteristics, and learns from failures through curiosity-driven reflection. The system achieves this through tight integration with Honcho, a memory infrastructure that enables persistent learning across tasks and meta-cognitive awareness of solving strategies.

## 1. Introduction

### 1.1 Problem Context

The ARC-AGI benchmark presents abstract reasoning challenges where solvers must infer transformation rules from a small number of input-output examples and apply these rules to novel test cases. Traditional approaches treat this as a pattern matching or rule induction problem, applying pre-defined strategies until one succeeds. This mechanical approach lacks the meta-cognitive awareness that characterizes human problem-solving.

### 1.2 Core Innovation

Arceus introduces a meta-cognitive framework where the solver reasons about its own reasoning process. Before attempting to solve a puzzle, it asks: "What type of problem is this? How should I think about it? What past experiences are relevant?" After failures, it reflects: "Why did this fail? What assumptions were wrong? What alternative interpretations exist?" This meta-level reasoning is enabled by Honcho, which provides both persistent memory and a dialectic interface for reflective queries.

### 1.3 System Architecture

Arceus operates through three interconnected cognitive layers that work in concert:

**Layer 1: Meta-Strategy Planning** decides how to think about the problem before attempting solutions.

**Layer 2: Adaptive Memory System** queries past experiences in a problem-type-specific manner.

**Layer 3: Curiosity-Driven Reflection** explores alternative interpretations after failures.

These layers surround a core execution engine that implements six solving strategies, all informed by the meta-cognitive insights from the three layers.

## 2. Honcho Memory Infrastructure

### 2.1 Memory Model

Honcho provides a structured memory system based on workspaces, peers, and sessions. In Arceus's deployment:

**Workspace** represents the entire ARC-AGI solving domain. All memory operations occur within a single workspace that persists across all solving sessions.

**Peers** represent different cognitive agents within the system. Arceus uses three primary peers: a solution generator peer for strategy execution, a reflection peer for meta-cognitive operations, and a dialectic peer for conversational memory queries.

**Sessions** group related memories together. Arceus maintains separate sessions for self-play exploration, meta-strategy learnings, curiosity insights, and primitive discoveries.

**Messages** are the atomic units of memory. Each message contains content, metadata, and timestamps. Messages can represent strategy results, meta-strategies, curiosity reflections, failure patterns, or discovered primitives.

### 2.2 Dialectic Interface

The dialectic API enables natural language queries against accumulated memory. Rather than performing vector similarity search directly, Arceus poses questions to Honcho such as "What spatial transformation strategies have worked on similar puzzles?" or "What assumptions commonly lead to failure in compositional problems?" Honcho processes these queries using language model reasoning over the stored message history, returning synthesized insights rather than raw retrieval results.

This dialectic approach enables several key capabilities:

**Semantic Retrieval**: Queries can be abstract and conceptual rather than keyword-based.

**Synthesis**: Multiple related memories are combined into coherent insights.

**Context-Awareness**: Query results consider the current problem context and past query patterns.

**Meta-Cognitive Reflection**: The system can query its own learning process, asking questions like "What have I learned about how to learn?"

### 2.3 Memory Operations

Arceus performs four primary memory operations:

**Ingestion** stores new experiences. After each puzzle attempt, Arceus stores the analysis, meta-strategy, memory reflections, strategy results, and curiosity insights. Each item is tagged with metadata including problem type, approach type, success status, and temporal information.

**Retrieval** queries relevant past experiences. Queries are constructed based on the current problem's characteristics and the meta-strategy's guidance. Retrieval is not a simple lookup but a dialectic conversation that synthesizes relevant insights.

**Reflection** reasons about stored memories. Rather than directly querying for similar puzzles, Arceus asks meta-questions about patterns in its memory: "For spatial problems, which approaches tend to work? Which assumptions tend to fail?"

**Adaptation** modifies query strategies based on problem type. The same puzzle patterns might trigger different memory queries depending on whether the meta-strategy classifies it as spatial reasoning versus pattern recognition.

## 3. Three-Layer Meta-Cognitive Architecture

### 3.1 Layer 1: Meta-Strategy Planning

Before attempting any solution strategy, Arceus engages in meta-strategy planning. This layer addresses a fundamental question: given a puzzle's characteristics, how should I think about solving it?

#### 3.1.1 Problem Type Classification

The meta-strategy planning process begins by analyzing the puzzle to determine its fundamental cognitive type. Rather than relying on predefined categories, this classification emerges from dialectic reasoning about the puzzle's characteristics.

The system considers:

**Pattern Recognition Problems** exhibit visual or structural patterns that must be identified and extended. The core challenge is recognizing what remains invariant versus what changes.

**Spatial Reasoning Problems** involve objects, positions, and spatial transformations. Success requires understanding geometric relationships and how objects move or transform in space.

**Logical Rule Problems** are governed by conditional or sequential rules that must be induced from examples. The challenge is discovering the rule structure rather than visual patterns.

**Compositional Problems** require breaking down complex transformations into simpler sub-transformations that compose to produce the final result.

Most puzzles exhibit characteristics of multiple types. Meta-strategy planning identifies the dominant type and any secondary characteristics, producing classifications like "spatial reasoning with compositional elements" or "pattern recognition with logical constraints."

#### 3.1.2 Thinking Strategy Selection

Based on problem type, meta-strategy planning selects an appropriate thinking approach:

**Analytical Thinking** proceeds systematically, decomposing the problem into parts, analyzing each part, and synthesizing a solution. This approach suits logical rule problems and complex compositional tasks.

**Intuitive Thinking** relies on pattern recognition and analogical reasoning, matching the current puzzle to similar past experiences. This approach works well for pattern recognition problems and familiar puzzle types.

**Experimental Thinking** tries variations and learns from results, iteratively refining hypotheses. This approach suits novel or ambiguous puzzles where the pattern is not immediately apparent.

**Compositional Thinking** breaks complex problems into simpler sub-problems, solves each independently, then combines solutions. This approach is essential for puzzles involving multiple transformation steps.

**Analogical Thinking** identifies structural similarities to past problems, adapting known solutions to the current context. This approach leverages past experience effectively.

The selected thinking strategy influences how subsequent layers query memory and how the execution engine sequences attempts.

#### 3.1.3 Memory Query Strategy Formulation

Meta-strategy planning determines how Layer 2 should query memory. Rather than generic queries, the strategy specifies:

**Query Focus**: What aspects of past experience are relevant? For spatial problems, focus on transformation types and object relationships. For pattern recognition, focus on visual patterns and matching strategies.

**Query Depth**: Should retrieval focus on similar puzzles or broader problem-type patterns? Novel problems benefit from broader retrieval, while familiar types need specific precedents.

**Query Composition**: Should multiple query types be combined? Compositional problems might query both sub-problem patterns and composition strategies.

**Temporal Weighting**: How much to favor recent versus historical memories? Recent successes often indicate effective current strategies, but historical patterns reveal stable problem-type knowledge.

#### 3.1.4 Assumption Identification

A critical meta-strategy function is identifying implicit assumptions that might constrain thinking. Common assumptions include:

**Transformation Uniformity**: Assuming the same transformation applies everywhere, when it might be context-dependent.

**Object Independence**: Assuming objects transform independently, when relationships between objects might matter.

**Single-Step Processing**: Assuming one transformation suffices, when multiple steps might compose.

**Global Patterns**: Assuming patterns are global, when they might be local or region-specific.

Explicitly identifying these assumptions enables Layer 3 to question them after failures, leading to paradigm shifts that unlock solutions.

#### 3.1.5 Curiosity Question Generation

Meta-strategy planning generates curiosity questions that guide exploration:

"What if the transformation depends on spatial context rather than being uniform?"

"What if object relationships matter more than individual object properties?"

"What if this requires multiple composition steps rather than a single transformation?"

These questions prime Layer 3's reflection process and help direct attention during strategy execution.

### 3.2 Layer 2: Adaptive Memory System

Layer 2 implements adaptive memory retrieval, where query strategies adjust based on the meta-strategy from Layer 1. This adaptation ensures retrieved memories are relevant to the current problem type and thinking approach.

#### 3.2.1 Query Construction Algorithm

Adaptive query construction proceeds through several stages:

**Base Context Formation**: The query begins with the puzzle's observed patterns and characteristics, establishing common ground between the current problem and memory.

**Problem-Type Adaptation**: Based on meta-strategy classification, the query is adapted to focus on type-specific aspects. For spatial reasoning, queries emphasize transformation types, object relationships, and geometric properties. For pattern recognition, queries focus on visual patterns, repetition, and symmetry. For logical rules, queries target conditional structures and rule discovery methods.

**Approach-Type Prioritization**: The query further adapts based on the selected thinking approach. Analytical queries emphasize causal understanding and ask "why" certain strategies worked. Intuitive queries emphasize empirical patterns and ask "what" strategies succeeded. Experimental queries focus on discovery processes and what experiments revealed insights. Compositional queries prioritize multi-step solutions and sub-problem patterns. Analogical queries seek structurally similar problems rather than superficially similar ones.

**Strategy Guidance Integration**: The meta-strategy's specific guidance is incorporated, directing attention to particular aspects of past experience. This might emphasize avoiding known failure patterns, exploring untried approaches, or building on successful precedent.

**Structured Response Specification**: The query explicitly requests structured information including summary insights, successful strategies with problem-type context, failure patterns with root causes, key insights for this problem type, and meta-commentary on how the query was adapted.

#### 3.2.2 Memory Retrieval Process

Retrieval occurs through Honcho's dialectic interface rather than direct database queries. The constructed query is posed as a natural language question to the reflection peer, which has access to all accumulated memories.

The dialectic process:

**Semantic Interpretation**: Honcho interprets the query's intent, understanding that a question about "spatial transformations that worked" seeks not just mentions of transformations but their contexts and success patterns.

**Relevant Memory Selection**: Rather than vector similarity alone, selection considers metadata tags, success patterns, recency, and relevance to the query's problem type and approach.

**Synthesis and Summarization**: Multiple related memories are synthesized into coherent insights rather than returned as separate fragments. This synthesis identifies patterns across memories, resolves contradictions, and formulates actionable guidance.

**Meta-Commentary**: The response includes meta-information about the retrieval process itself, noting how the query was interpreted, what adaptations were applied, and what types of memories contributed to the response.

#### 3.2.3 Retrieval Adaptation Examples

To illustrate adaptive retrieval, consider how the same puzzle patterns trigger different queries based on problem type:

For a puzzle with rotation patterns classified as spatial reasoning with compositional approach:

Query focuses on multi-step spatial transformations, how problems decomposed successfully in the past, which object relationships mattered, and what compositional assumptions failed. The response prioritizes strategies that decompose spatial problems, emphasizes object-relationship analysis, and warns against assuming independent transformations.

For a puzzle with similar rotation patterns classified as pattern recognition with intuitive approach:

Query focuses on what visual patterns matched similar puzzles, which pattern-matching strategies succeeded empirically, and what quick heuristics worked. The response emphasizes visual similarity matches, provides empirical success patterns, and suggests analogical reasoning from similar puzzles.

This adaptation ensures retrieved memories align with how the solver intends to think about the problem.

#### 3.2.4 Failure Pattern Integration

Adaptive memory specifically queries for failure patterns relevant to the current problem type. Rather than generic "strategies that failed," queries ask:

"For compositional spatial problems, what assumptions about object independence commonly fail?"

"For pattern recognition with complex visual structure, what simplifying assumptions prove wrong?"

"For logical rule problems, what conditional structures are commonly missed?"

This targeted failure retrieval helps avoid repeating type-specific mistakes.

### 3.3 Layer 3: Curiosity-Driven Reflection

After strategy failures, Layer 3 engages curiosity-driven reflection to explore why failure occurred and what alternative interpretations might succeed. This layer transforms failures from dead ends into learning opportunities.

#### 3.3.1 Root Cause Analysis

Curiosity reflection begins by analyzing the failure's root cause. Rather than superficial "the output didn't match," this analysis examines:

**Assumption Violations**: Which implicit assumptions from Layer 1 were violated? If the solver assumed uniform transformation but the pattern is context-dependent, that assumption caused failure.

**Incomplete Patterns**: Were observed patterns incomplete or misleading? Perhaps a pattern holds for training examples but breaks in the test case.

**Wrong Abstraction Level**: Was the solver reasoning at the wrong level of abstraction? Perhaps it focused on pixel-level patterns when object-level relationships matter, or vice versa.

**Missed Relationships**: Did the solver miss critical relationships between elements? Perhaps objects interact in ways that weren't apparent from individual object analysis.

**Incorrect Composition**: For multi-step problems, was the composition order or method wrong? Perhaps transformations must apply in specific sequences or with specific interactions.

Root cause analysis draws on the meta-strategy's identified assumptions and the actual failure evidence to pinpoint where reasoning went wrong.

#### 3.3.2 Alternative Interpretation Generation

The core of curiosity reflection is generating alternative interpretations of the puzzle. This process asks "what if" questions that challenge the current interpretation:

**What if spatial context matters more than assumed?** Perhaps transformations aren't uniform but depend on position, neighborhood, or region.

**What if objects have hidden relationships?** Perhaps objects that appear independent actually influence each other, with transformations depending on relative positions or properties.

**What if the pattern is compositional rather than atomic?** Perhaps what appears as a single complex transformation is actually multiple simpler transformations composed.

**What if the abstraction level is wrong?** Perhaps reasoning about pixels when object-level reasoning is needed, or reasoning about objects when relationship-level reasoning is needed.

**What if temporal sequencing matters?** Perhaps transformations must apply in specific orders, with earlier transformations creating contexts for later ones.

These alternative interpretations directly challenge the assumptions that led to failure, opening new solution paths.

#### 3.3.3 Blind Spot Identification

Curiosity reflection explicitly identifies blind spots—aspects of the puzzle that were overlooked or dismissed. Common blind spots include:

**Spatial Context Effects**: Assuming properties are intrinsic to objects when they're actually contextual, determined by surrounding elements or global position.

**Interaction Effects**: Missing that objects influence each other's transformations rather than transforming independently.

**Multi-Scale Patterns**: Focusing on one scale while missing patterns at other scales, such as noticing local patterns but missing global structure or vice versa.

**Negative Space**: Ignoring background or empty space, which might carry semantic meaning or structural constraints.

**Implicit Ordering**: Missing that elements have implicit orderings (spatial, temporal, or semantic) that affect transformations.

Identifying blind spots helps direct attention to previously ignored aspects during subsequent attempts.

#### 3.3.4 Paradigm Shift Formulation

When curiosity reflection reveals fundamental interpretation errors, it formulates paradigm shifts—fundamentally different ways to conceptualize the problem:

**From uniform to contextual**: Stop treating transformations as uniform; start considering how context modulates transformation.

**From independent to relational**: Stop analyzing objects independently; start reasoning about relationship graphs and interaction patterns.

**From atomic to compositional**: Stop seeking single transformations; start decomposing into sequences of simpler transformations.

**From pixel-based to object-based**: Stop reasoning about pixel patterns; start identifying objects and reasoning about object-level transformations.

**From object-based to relationship-based**: Stop reasoning about objects; start reasoning about relationships between objects as first-class entities.

Paradigm shifts don't just suggest trying a different strategy; they reframe the entire problem interpretation, often unlocking solutions that were impossible under the previous interpretation.

#### 3.3.5 Experimental Idea Generation

Curiosity reflection generates specific experiments that could test hypotheses or explore alternative interpretations:

"Try treating transformations as context-dependent, with transformation type determined by spatial region."

"Try analyzing pairs of objects for relationship patterns rather than objects individually."

"Try decomposing the observed complex transformation into sequences of simpler primitive operations."

"Try identifying objects first, then reasoning about object-level transformations rather than pixel-level patterns."

These experimental ideas guide subsequent strategy attempts, ensuring they explore the alternative interpretations identified through curiosity.

## 4. Self-Play Exploration

Self-play exploration is Arceus's primary learning mechanism, where the system autonomously explores ARC-AGI puzzles to build up memory and discover effective solving strategies.

### 4.1 Exploration Loop

The self-play exploration loop operates as follows:

**Task Selection**: A training task is selected for exploration. Task selection can be random, sequential, or targeted toward specific problem types to build balanced experience.

**Initial Analysis**: The puzzle is analyzed to extract observable patterns including color usage, grid sizes, object counts, symmetries, spatial relationships, and transformation hints from comparing input-output pairs.

**Meta-Strategy Planning**: Layer 1 engages, classifying the problem type, selecting a thinking approach, planning memory query strategy, identifying assumptions, and generating curiosity questions. All meta-strategy decisions are stored in memory for future reference.

**Adaptive Memory Reflection**: Layer 2 queries memory using the meta-strategy's guidance, retrieving relevant past experiences adapted to the current problem type and approach. Retrieved insights inform strategy selection and execution.

**Strategy Execution**: Six solving strategies are attempted in sequence. Each attempt is informed by meta-strategy guidance and memory insights. Strategies tried include code generation, primitive combinations, pattern matching, code mutation, hybrid approaches, and creative combinations.

**Outcome Recording**: After each strategy attempt, the result (success or failure), execution details, and observable effects are recorded. Successful strategies are stored with full context. Failed attempts undergo curiosity reflection.

**Curiosity Reflection**: After failures, Layer 3 engages to analyze root causes, generate alternative interpretations, identify blind spots, formulate paradigm shifts, and generate experimental ideas. All curiosity insights are stored for future use.

**Primitive Discovery**: If no strategy succeeds, the system attempts to invent new primitive transformations based on observed input-output patterns. Discovered primitives are stored in memory for use in future puzzles.

**Memory Storage**: All learnings from the exploration are ingested into memory including meta-strategies that worked or failed, memory retrieval adaptations that were helpful, curiosity insights that led to breakthroughs, strategy results with full context, failure patterns and their root causes, and discovered primitives with usage contexts.

### 4.2 Strategy Sequencing

Strategy attempts are sequenced intelligently based on meta-strategy guidance and memory insights:

**Strategy Reasoning**: Before attempting the first strategy, if sufficient memory exists, the system reasons about strategy effectiveness using dialectic queries. This might reveal "code generation rarely works for highly visual pattern problems" or "primitive combinations are effective for spatial transformations."

**Dynamic Reordering**: Based on reasoning results, strategy order might be adjusted. If memory suggests a particular strategy type is effective for the current problem type, that strategy moves to front.

**Attempt Limit Adaptation**: The system dynamically adjusts how many strategies to try based on problem characteristics and past success patterns. Simple problems might warrant fewer attempts, while complex problems justify exhaustive exploration.

**Informed Execution**: Each strategy execution receives context from meta-strategy planning and memory reflection, enabling strategies to specialize their approach based on problem type and past learnings.

### 4.3 Cross-Task Learning

Self-play enables cross-task learning through memory accumulation. Early in exploration, the system lacks type-specific knowledge and tries strategies mechanically. As memory accumulates:

**Problem Type Patterns**: The system learns "for spatial reasoning problems, compositional thinking works well" and "pattern recognition problems benefit from intuitive approaches."

**Strategy Effectiveness Patterns**: Memory reveals which strategies tend to work for which problem types, enabling better strategy selection.

**Failure Pattern Avoidance**: Accumulated failure memories help avoid repeating type-specific mistakes like "assuming object independence in compositional problems" or "missing spatial context in transformation problems."

**Meta-Strategy Refinement**: The system learns not just what strategies work, but how to think about different problem types, refining meta-strategy planning over time.

**Primitive Library Growth**: As new primitives are discovered across different tasks, the primitive library expands, enabling more complex transformations through composition.

This cross-task learning means later explorations benefit from all previous experience, with the system becoming increasingly effective as memory accumulates.

## 5. Solving Strategies

Arceus implements six distinct solving strategies, each taking different approaches to inferring and applying transformations.

### 5.1 Code Generation Strategy

The code generation strategy attempts to synthesize a Python function that transforms inputs to outputs based on training examples.

#### 5.1.1 Generation Process

Code generation proceeds through dialectic interaction:

**Training Example Analysis**: The system analyzes input-output pairs from training examples, identifying patterns, invariants, and transformation characteristics.

**Memory-Guided Generation**: Using the meta-strategy's problem type and retrieved memory insights, the system queries for similar past transformations and successful code patterns for this problem type.

**Code Synthesis**: Through dialectic interaction, a Python transformation function is generated that attempts to capture the observed pattern. The generated code includes numpy array operations for grid manipulation, conditional logic for pattern-dependent behavior, and iterative processing for spatial transformations.

**Validation on Training Examples**: The generated code is executed on training examples to verify it produces correct outputs. Execution occurs in a sandboxed environment with timeout protection.

**Refinement Loop**: If training example validation fails, the system analyzes failure modes, requests code refinements through dialectic interaction, and validates refined code. This refinement loop continues for up to five iterations.

**Test Application**: Once code passes validation on all training examples, it's applied to the test input to generate the solution.

#### 5.1.2 Memory Integration

Code generation integrates memory in several ways:

**Pattern Library Access**: Retrieves code patterns that worked for similar problem types from memory, providing templates and approaches.

**Failure Pattern Avoidance**: Queries memory for common code generation failures on this problem type, such as "forgetting to handle edge cases in spatial transformations" or "assuming uniform color mapping when it's context-dependent."

**Refinement Guidance**: During refinement loops, memory insights about why similar code failed guide debugging and fixing approaches.

### 5.2 AIRV Augmentation Strategy

AIRV (Augment-Inference-Reverse-Vote) is a strategy that generates multiple augmented versions of the problem, solves each, reverses augmentations, and votes on the most common result.

#### 5.2.1 Augmentation Generation

The system generates augmented versions of the puzzle through transformations that should be invariant:

**Rotation Augmentation**: Creates rotated versions (90°, 180°, 270°) under the assumption that transformation rules are rotation-invariant.

**Reflection Augmentation**: Creates horizontally and vertically flipped versions under the assumption that transformation rules are reflection-invariant.

**Color Permutation**: Creates versions with permuted color mappings under the assumption that specific color values don't matter, only their relative distinctions.

Not all augmentations are valid for all problems. The system uses dialectic reasoning to determine which augmentations are likely to preserve the core transformation pattern.

#### 5.2.2 Inference and Reverse Process

For each augmented version:

**Augmented Solving**: A base solver (typically code generation or primitive combination) attempts to solve the augmented puzzle using augmented training examples and augmented test input.

**Reverse Transformation**: If a solution is obtained, the augmentation is reversed. A solution to a rotated puzzle is rotated back; a solution to a color-permuted puzzle has color permutations reversed.

**Result Collection**: All reversed solutions are collected for voting.

#### 5.2.3 Voting Mechanism

The voting process determines the final solution:

**Frequency Counting**: Each unique solution grid is counted across all augmentation results.

**Majority Selection**: The most frequently occurring solution is selected as the final answer.

**Confidence Assessment**: The vote distribution provides confidence information. Unanimous agreement indicates high confidence; split votes suggest uncertainty.

AIRV is particularly effective for problems where the transformation is fundamentally invariant to certain augmentations, as it effectively averages out errors and biases from individual solving attempts.

### 5.3 Primitive Combination Strategy

The primitive combination strategy solves puzzles by composing sequences of primitive transformations.

#### 5.3.1 Primitive Library

Arceus maintains a library of primitive transformations including:

**Geometric Primitives**: Rotation (90°, 180°, 270°), reflection (horizontal, vertical), transposition, shift operations (up, down, left, right).

**Color Primitives**: Color swapping, color replacement, fill operations, background extraction.

**Object Primitives**: Object extraction, object filtering, object property analysis, object relationship analysis.

**Pattern Primitives**: Pattern repetition, pattern extraction, symmetry operations, scaling operations.

Beyond built-in primitives, the library grows through primitive discovery, where new primitives are invented during self-play exploration.

#### 5.3.2 Combination Search

Finding effective primitive combinations involves:

**Hypothesis Generation**: Based on pattern analysis and memory insights, hypotheses about useful primitive sequences are generated. For instance, "extract red objects, rotate 90°, place in grid" or "find largest object, flip horizontally, replicate."

**Compositional Search**: Primitives are tried in various combinations and sequences. Search is guided by meta-strategy insights about problem type and memory patterns about effective compositions.

**Validation**: Each combination is tested on training examples. Combinations that successfully transform all training inputs to their corresponding outputs are candidates for test application.

**Test Application**: The validated primitive combination is applied to the test input to generate the solution.

#### 5.3.3 Memory-Guided Composition

Memory integration makes primitive composition more efficient:

**Effective Composition Patterns**: Memory stores which primitive combinations worked for which problem types, focusing search on likely combinations.

**Failure Patterns**: Memory stores which combinations tend to fail for which problem types, avoiding unproductive paths.

**Composition Strategies**: Memory accumulates meta-knowledge about composition strategies, such as "for spatial problems, start with object extraction" or "for pattern problems, try symmetry operations early."

### 5.4 Pattern Matching Strategy

Pattern matching attempts to match the current puzzle to similar past puzzles and adapt their solutions.

#### 5.4.1 Similarity Assessment

Pattern matching begins by assessing similarity to past puzzles in memory:

**Structural Similarity**: Comparing grid dimensions, object counts, color usage, and spatial arrangements.

**Transformation Similarity**: Comparing observed transformation characteristics from training examples to past transformation patterns.

**Problem Type Similarity**: Using meta-strategy classification to find puzzles of the same type.

**Pattern Similarity**: Comparing specific patterns like symmetries, repetitions, or spatial relationships.

#### 5.4.2 Solution Adaptation

When similar puzzles are found:

**Solution Retrieval**: The solving approach that worked for the similar puzzle is retrieved from memory.

**Adaptation Analysis**: Differences between the current puzzle and the retrieved puzzle are analyzed to determine needed adaptations.

**Solution Modification**: The retrieved solution approach is modified to account for differences, adjusting parameters, transformations, or composition sequences.

**Validation and Application**: The adapted solution is validated on training examples before applying to the test case.

### 5.5 Code Mutation Strategy

Code mutation builds on the code generation strategy by creating variations of generated code.

#### 5.5.1 Mutation Operations

When code generation produces incorrect results, mutation creates variations:

**Parameter Mutation**: Modifying numerical parameters in the code, such as rotation angles, shift amounts, or threshold values.

**Operation Substitution**: Replacing operations with related operations, such as swapping rotation direction or changing comparison operators.

**Conditional Mutation**: Modifying conditional logic, such as changing condition orders or adding additional checks.

**Composition Mutation**: Reordering transformation steps or adding intermediate steps.

#### 5.5.2 Guided Mutation

Curiosity reflection from Layer 3 guides mutation:

**Root Cause-Based Mutation**: If curiosity identified that "uniform transformation assumption failed," mutations introduce context-dependent behavior.

**Alternative Interpretation-Based Mutation**: If curiosity suggested "objects might interact," mutations add relationship-checking logic.

**Paradigm Shift-Based Mutation**: If curiosity proposed a paradigm shift, mutations reframe the entire code structure accordingly.

### 5.6 Hybrid and Creative Strategies

Hybrid strategies combine multiple approaches:

**Code-Guided Primitive Search**: Using generated code to identify promising primitive operations, then searching for primitive combinations that match the code's logic.

**AIRV with Multiple Base Solvers**: Running AIRV with different base solvers for augmented puzzles, providing more diverse votes.

**Memory-Informed Code Generation**: Directly incorporating retrieved code patterns from memory into generated code rather than starting from scratch.

Creative strategies attempt novel combinations and approaches based on curiosity insights and memory patterns about what has been untried.

## 6. Complete Algorithm Flow

The complete Arceus algorithm integrates all components described above. Here we present the algorithm flow for solving a single puzzle:

### 6.1 Problem Receipt and Initial Analysis

Upon receiving a puzzle, Arceus extracts observable characteristics from the training examples. This analysis identifies color distributions across inputs and outputs, grid size patterns and changes, object counts and their evolution, spatial patterns including symmetries and repetitions, apparent transformations visible from comparing examples, and relational patterns between elements.

This initial analysis provides the foundation for meta-strategy planning, informing problem type classification and pattern-based memory queries.

### 6.2 Meta-Strategy Planning Phase

Before attempting any solution, Layer 1 engages in meta-strategy planning through dialectic interaction with Honcho's reflection peer.

The planning query provides the observed patterns and asks for reasoning about problem type classification, appropriate thinking approach, how memory should be queried given this problem type, what assumptions commonly lead to failure for this type, what curiosity questions should guide exploration, and how strategies should be sequenced for this type.

The response from this dialectic interaction provides a complete meta-strategy that guides all subsequent phases. This meta-strategy is immediately stored in memory for future reference, tagged with the problem's characteristics and metadata about the classification process.

### 6.3 Adaptive Memory Reflection Phase

Layer 2 constructs an adaptive memory query based on the meta-strategy. The query construction follows the algorithm described in Section 3.2.1, producing a problem-type-specific and approach-specific query.

This query is posed to Honcho's reflection peer, which synthesizes relevant memories and returns structured insights including summary recommendations, successful strategies for this problem type, failure patterns to avoid, untried approaches worth considering, and meta-commentary on how the query was adapted.

The memory reflection results are used to inform strategy selection and execution. Strategies are potentially reordered based on memory-informed effectiveness predictions.

### 6.4 Strategy Execution Phase

Strategies are attempted in sequence, each receiving context from meta-strategy planning and memory reflection.

For each strategy:

**Strategy Initialization**: The strategy is initialized with meta-strategy guidance, memory insights, and problem characteristics.

**Execution with Monitoring**: The strategy executes, generating a candidate solution. Execution is monitored for resource usage, timeout protection, and error handling.

**Validation**: The candidate solution is validated against training examples. Full success requires matching all training outputs.

**Result Recording**: Strategy results are recorded including success or failure status, execution details, generated outputs, and notable features of the attempt.

**Success Termination**: If validation succeeds, the strategy's approach is applied to the test input to generate the final solution. All context about how success was achieved is stored in memory.

**Failure Processing**: If validation fails, the attempt proceeds to Layer 3 for curiosity reflection before trying the next strategy.

### 6.5 Curiosity Reflection Phase

After each strategy failure, Layer 3 engages to transform the failure into learning.

The curiosity reflection process receives the failed strategy's details, the meta-strategy context, the memory insights, and the failure evidence (generated outputs versus expected outputs).

Through dialectic interaction, Layer 3 analyzes why the failure occurred, what assumptions were violated, what alternative interpretations might work, what blind spots might have been missed, what paradigm shifts might unlock success, and what experiments could test alternative hypotheses.

The curiosity insights are immediately stored in memory and used to inform subsequent strategy attempts. Later strategies can adapt based on curiosity-discovered alternative interpretations, avoiding the same failure modes.

### 6.6 Primitive Discovery Phase

If all strategies fail during self-play exploration, the system attempts to discover new primitives that might capture the transformation pattern.

Primitive discovery analyzes the consistent transformations across training examples, searches for compositional patterns that might be captured as reusable operations, generates candidate primitive implementations through dialectic code synthesis, validates candidates on training examples, and stores validated primitives in memory for future use.

Discovered primitives expand the system's capability space, enabling future puzzles to be solved through primitive combinations.

### 6.7 Memory Ingestion Phase

After solving or exhausting attempts, all learnings are systematically ingested into memory.

Meta-strategy information is stored including the problem type classification, selected thinking approach, memory query strategy, and effectiveness assessment.

Memory reflection data is stored including the adaptive query used, insights retrieved, how retrieval was adapted, and usefulness assessment.

Strategy results are stored including which strategies were attempted, success or failure of each, execution details, and contextual factors affecting outcomes.

Curiosity reflections are stored including root cause analyses, alternative interpretations generated, blind spots identified, paradigm shifts proposed, and whether curiosity insights led to eventual success.

Failure patterns are stored including anti-patterns observed, common mistakes for this problem type, and recommendations for avoiding similar failures.

Successful approaches are stored including the complete solution path, why it worked, what made it effective for this problem type, and reusable patterns for similar problems.

Each memory item is tagged with rich metadata enabling future retrieval including problem type, approach type, timestamp, success status, strategy types involved, and thematic tags.

### 6.8 Cross-Task Learning Accumulation

As exploration proceeds across multiple tasks, memory accumulates patterns that transcend individual puzzles:

**Problem Type Knowledge**: The system learns generalizations about problem types, such as "spatial reasoning problems benefit from compositional decomposition" or "pattern recognition problems often have local-to-global structure."

**Strategy Effectiveness Patterns**: Memory reveals which strategies tend to work for which problem types, enabling progressively better strategy selection as experience grows.

**Meta-Learning**: The system learns about its own learning process, discovering patterns like "curiosity insights about object relationships frequently lead to breakthroughs" or "meta-strategies that emphasize compositional thinking succeed on complex puzzles."

**Failure Pattern Libraries**: Accumulated failure memories create libraries of anti-patterns for each problem type, enabling proactive avoidance of known pitfalls.

**Primitive Composition Knowledge**: As primitives are discovered and used across tasks, memory accumulates knowledge about effective composition patterns, learning which primitives combine well and for what problem types.

This accumulated meta-knowledge enables increasingly sophisticated problem-solving, with later tasks benefiting from all previous experience synthesized through Honcho's memory system.

## 7. Key Algorithm Properties

### 7.1 Meta-Cognitive Awareness

Unlike traditional solvers that mechanically try strategies, Arceus maintains awareness of its own reasoning process. Before acting, it reasons about how to think. After failures, it reasons about why its thinking failed. This meta-cognitive awareness enables the system to adapt its approach based on problem characteristics rather than blindly applying fixed strategies.

### 7.2 Adaptive Intelligence

The three-layer architecture ensures adaptation at multiple levels. Meta-strategy planning adapts the overall approach to problem type. Adaptive memory retrieval adapts queries to retrieve type-relevant experiences. Curiosity reflection adapts interpretation based on failure patterns. This multi-level adaptation enables flexible problem-solving across diverse puzzle types.

### 7.3 Learning from Failure

Traditional solvers treat failures as dead ends, moving to the next strategy. Arceus treats failures as learning opportunities, engaging curiosity reflection to understand what went wrong and what alternatives exist. This transforms failures into knowledge that improves future attempts on the current puzzle and future puzzles of similar types.

### 7.4 Knowledge Transfer

Through Honcho's memory system, knowledge transfers across puzzles. Learnings about problem types, strategy effectiveness, failure patterns, and meta-strategies accumulate and apply to new problems. This enables the system to become progressively more effective as it gains experience.

### 7.5 Dialectic Reasoning

Rather than treating memory as a lookup table, Arceus engages in dialectic reasoning with Honcho, posing questions and synthesizing answers. This enables semantic, context-aware retrieval that adapts to the current problem's characteristics, producing insights rather than raw data.

## 8. Architectural Decisions and Rationale

### 8.1 Three-Layer Separation

The separation into three distinct layers (meta-strategy, adaptive memory, curiosity) serves several purposes:

**Separation of Concerns**: Each layer addresses a different aspect of cognition—planning, experience retrieval, and reflection—enabling each to be sophisticated without interfering with others.

**Modularity**: Layers can be developed, tested, and improved independently while maintaining clean interfaces.

**Transparency**: The three-layer structure makes the system's reasoning explicit and observable, supporting debugging, analysis, and trust.

**Composability**: Insights from one layer inform others in a structured flow, enabling complex meta-cognitive behaviors from simple layer interactions.

### 8.2 Honcho Integration Choice

Integrating with Honcho rather than implementing custom memory provides several advantages:

**Persistent Memory**: Honcho provides durable storage that persists across sessions, enabling long-term learning.

**Dialectic Interface**: Honcho's dialectic API enables natural language queries and synthesis, supporting complex semantic retrieval that would be difficult with vector search alone.

**Multi-Peer Architecture**: Honcho's peer model enables multiple specialized agents (solution generator, reflection, dialectic) with shared memory, supporting distributed cognition.

**Structured Metadata**: Honcho's message model with metadata enables rich tagging and filtering, supporting problem-type-specific retrieval.

**Scalability**: Honcho handles memory management, indexing, and retrieval optimization, allowing Arceus to focus on reasoning rather than infrastructure.

### 8.3 Dialectic Over Direct Retrieval

The choice to query memory through dialectic interaction rather than direct vector search enables:

**Semantic Understanding**: Queries can be abstract and conceptual, asking about "why strategies work" rather than matching keywords.

**Synthesis**: Multiple memories are combined into coherent insights rather than returned as fragments requiring assembly.

**Context-Awareness**: Retrieval considers the query's context and purpose, returning relevant insights rather than merely similar content.

**Adaptability**: Query interpretation can evolve as the language model improves, without changing Arceus's code.

### 8.4 Problem-Type Classification

Rather than hand-coded rules, problem type classification emerges from dialectic reasoning about puzzle characteristics. This approach:

**Flexibility**: New problem types can be recognized without code changes, as classification is reasoning-based rather than rule-based.

**Nuance**: Problems can be classified as combinations of types (e.g., "spatial with compositional elements"), capturing nuanced characteristics.

**Evolution**: As more puzzles are explored, classification reasoning improves through accumulated examples.

**Alignment**: Classification directly supports the goal of adaptive memory—determining what type of past experience is relevant.

## 9. Limitations and Future Directions

### 9.1 Current Limitations

**Computational Cost**: The three-layer architecture involves multiple dialectic queries per puzzle, adding latency. Each layer requires language model inference, accumulating to approximately 500-800ms overhead per puzzle.

**Memory Dependency**: Effectiveness depends on accumulated memory. Early in exploration with sparse memory, the system lacks the experiential foundation for effective meta-strategy and adaptation.

**Strategy Coverage**: While six strategies cover many puzzle types, some puzzles may require approaches not represented, particularly those requiring specialized mathematical or logical reasoning.

**Prompt Sensitivity**: Dialectic queries depend on prompt formulation. Query phrasing affects response quality, requiring careful prompt engineering.

**Limited Meta-Learning**: While the system learns problem-type patterns, it has limited ability to reflect on its own meta-learning process or improve its learning strategies dynamically.

### 9.2 Potential Enhancements

**Dynamic Strategy Discovery**: Rather than fixed strategies, the system could discover new strategy types through exploration, expanding its capability space organically.

**Multi-Agent Collaboration**: Multiple Arceus instances could collaborate on difficult puzzles, sharing meta-strategies and trying different approaches in parallel.

**Hierarchical Memory**: Memory could be organized hierarchically, with working memory for current puzzle, episodic memory for recent explorations, and semantic memory for accumulated meta-knowledge.

**Active Learning**: Rather than random exploration, the system could strategically select which puzzles to explore based on learning value, focusing on problem types where knowledge is sparse.

**Meta-Meta-Learning**: The system could reflect on its own meta-learning patterns, discovering principles about when meta-strategies work and how to improve meta-strategy planning itself.

**Confidence Modeling**: Explicitly modeling confidence in solutions based on meta-strategy alignment, memory support, and voting patterns could enable selective application of computationally expensive techniques.

## 10. Conclusion

Arceus demonstrates that meta-cognitive reasoning—thinking about how to think—can significantly enhance abstract reasoning problem-solving. By integrating three cognitive layers (meta-strategy planning, adaptive memory retrieval, curiosity-driven reflection) with persistent memory provided by Honcho, the system achieves flexible, adaptive problem-solving that learns from experience across tasks.

The key insight is that the method of approach matters as much as the strategies attempted. Classifying problem types, adapting memory queries to types, and learning from failures through curiosity transforms solving from mechanical strategy application to intelligent reasoning that improves over time.

This architecture suggests broader principles for building adaptive AI systems: explicit meta-reasoning about task characteristics, experience-based adaptation of retrieval and reasoning, learning from failures as opportunities rather than dead ends, and persistent memory that accumulates and transfers knowledge across tasks.

The Arceus algorithm represents a step toward AI systems that don't just solve problems, but reason about how to solve them, learning not just what works but why it works and how to think about new challenges.
