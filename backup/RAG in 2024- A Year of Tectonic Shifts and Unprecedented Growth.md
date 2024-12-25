
2024 has been nothing short of a whirlwind for Retrieval-Augmented Generation (RAG). If the early murmurs of it being the "Year of RAG" weren't quite a universal consensus, the sheer volume of innovation and adoption throughout the year has certainly cemented its status as a pivotal force in the LLM landscape. Think of it like this: Large Language Models (LLMs) are the brilliant minds, but RAG is the invaluable research assistant, diligently fetching the right information at the right time.

To set the stage, consider this visual representation of the interconnected world of RAG (imagine the image you provided would be inserted here). It beautifully illustrates the various components working in concert.

This year witnessed several defining moments that shaped the trajectory of RAG. Let's delve into them:

### The Great RAG Debate:  Alive and Kicking!

The early days of LLMs saw a dichotomy: fine-tuning for specific tasks versus using external knowledge. In 2023 and before, the terminology around external knowledge was often informal – "external memory" or "external knowledge base." The debate centered around whether these "add-ons" were temporary fixes or if fine-tuning was the ultimate solution.

However, 2024 decisively tilted the scales. RAG's cost-effectiveness and real-time adaptability proved to be game-changers. While fine-tuning still has its place, RAG emerged as an indispensable component, even in scenarios where fine-tuning was necessary.

The first half of the year also saw another interesting debate emerge, fueled by the remarkable progress in LLM context windows. Could massive context windows negate the need for RAG? This sparked intense discussion, but by mid-year, a consensus formed: both have distinct strengths and are often complementary. Long context windows allow for processing larger documents directly, while RAG excels at retrieving relevant information from vast, dispersed knowledge bases.

### LLMOps Ascendant:  Building Blocks for Widespread Adoption

A significant catalyst for RAG's proliferation in 2024 was the maturity of LLMOps frameworks. These frameworks, visualized by the image mentioned earlier, provided a standardized blueprint for constructing RAG systems. Suddenly, businesses and individuals could assemble a robust RAG pipeline – connecting vector databases, embedding/reranker models, LLMs, text chunking strategies, and prompt management – with relative ease. This accessibility lowered the barrier to entry and fueled experimentation across various domains.

### Tackling the Core Challenges:  Beyond the Basics

While LLMOps provided the scaffolding, the real innovation in 2024 lay in addressing the fundamental limitations of early RAG implementations. Academic research (references [29], [30]) highlighted some traditional approaches, but practical experience revealed three key pain points:

* **The Multi-Modal Hurdle:**  Early RAG systems were primarily text-centric, struggling with non-structured, multi-modal documents like PDFs, presentations, and image-rich content. This left a vast amount of valuable corporate data untapped.
* **Vector Database Limitations:** Relying solely on vector databases often led to low recall and hit rates. Vector representations, while excellent for semantic similarity, can lack the precision needed for exact matches and sometimes suffer from semantic drift.
* **The Semantic Gap:** The core assumption of RAG – that a user's query will directly lead to the relevant document – often breaks down. Vague queries or "multi-hop" questions requiring synthesis of information from multiple sources were difficult to address effectively.

These challenges became the driving force behind the next wave of RAG innovation in 2024.

### Key Milestones of Innovation:

This year saw tangible progress in overcoming these limitations, marked by several significant developments:

**1. The Rise of Multi-Modal Document Parsing Tools:**  Recognizing the importance of non-textual data, a new breed of tools emerged, focusing on intelligently parsing PDFs, PPTs, and other complex document formats.

**2. BM25's Renaissance and the Hybrid Search Paradigm:**  The limitations of pure vector search became apparent, leading to a resurgence of traditional information retrieval techniques like BM25. Hybrid search, combining the semantic power of vector search with the precision of BM25 and other methods, became the dominant approach. The notion of a vector database as a standalone necessity began to fade.

**3. RAGFlow's Impact:  Setting New Standards:** The open-sourcing of the RAGFlow engine on April 1st, 2024 (reaching over 25,000 GitHub stars by year-end) was a watershed moment. Two key design principles within RAGFlow quickly gained widespread adoption:
    * **Semantic Chunking for Enhanced Data Quality:**  Instead of rudimentary text chunking, RAGFlow introduced semantic chunking for unstructured data. This involved using specialized models to understand document layout and structure, preventing simple splitters from disrupting meaningful content. The community embraced this, leading to a proliferation of similar tools.
    * **Hybrid Search as the Foundation:** RAGFlow championed enterprise-grade search engines with robust BM25 capabilities as the primary backend. This highlighted the enduring value of this classic algorithm for precise retrieval. While some vector databases added BM25 functionality (like the Qdrant "BM42" incident, later clarified as a misunderstanding), true enterprise-grade BM25 integration remained crucial. RAGFlow was instrumental in demonstrating the power of hybrid search and the diminishing need for standalone vector databases.

**4. GraphRAG Takes Center Stage:**  Microsoft's open-sourcing of GraphRAG was a phenomenon. As a library rather than a full product, its rapid adoption signaled its effectiveness in addressing the semantic gap. GraphRAG leverages knowledge graphs to bridge the disconnect between user queries and relevant information, enabling more complex and nuanced question answering.

**5. The Emergence of Late-Interaction Models (Col-xxx):**  These models marked a shift in how ranking was approached, moving towards more computationally efficient methods while still capturing crucial interaction information.

**6. Multi-Modal RAG Fueled by VLMs and Late-Interaction:**  The combination of powerful Vision Language Models (VLMs) and late-interaction models paved the way for true multi-modal RAG. This allowed for understanding and retrieval from documents containing images, charts, and other visual elements, unlocking significant business value.

These last two points highlighted the growing importance of sophisticated ranking and the need for databases with native tensor support. This led to innovations like those seen in the Infinity database, which incorporated these capabilities, even if they hadn't yet fully integrated into RAGFlow.

###  Deep Dive: Technological Advancements in RAG

Let's delve deeper into the technical innovations that powered RAG's progress in 2024. While academic surveys abound (references [27], [28], [38]), this section focuses on the practical implementation and impact of key research trends. We view RAG not as a simple application but as a complex ecosystem where search is central, and various data sources, components, and models collaborate.

**Data Cleaning:  Laying the Foundation for Quality**

The principle of "Quality In, Quality Out" is paramount for RAG. For multi-modal, unstructured documents, employing vision models for document layout analysis became crucial – a domain often referred to as Document Intelligence. While historically disparate, tasks like Table Structure Recognition (TSR) and image analysis for formulas and charts began to be integrated under a unified framework. This "generalized OCR" became the entry point for many RAG pipelines. RAGFlow's DeepDoc module was an early adopter of this approach, contributing to its initial popularity. Similar systems like MinuerU [2] and Docling [3] followed suit.

Document intelligence models evolved along two main paths:

* **Generation 1: Traditional Vision Models:**  These models, including the open-source RAGFlow DeepDoc module, were built on conventional vision architectures. Their advantage was CPU compatibility, but they often struggled with generalization across diverse scenarios, requiring specialized training for each use case – a process sometimes jokingly called "carving flowers."
* **Generation 2: Generative Models:**  The shift towards Transformer-based Encoder-Decoder architectures, exemplified by Meta's Nougat [4] and "OCR 2.0" [5], offered improved generalization. Structures like StructEqTable [6] directly applied similar networks for table reconstruction. RAGFlow's enterprise version also adopted this approach. While requiring more computational power, these models demonstrated superior performance across varied document types. A key advancement, as seen in M2Doc [23], was the integration of text information (like BERT) into vision-based encoders to better define semantic boundaries.

The trend towards Encoder-Decoder architectures for unified multi-modal document parsing is expected to accelerate in 2025.

Beyond multi-modal data, text-based document processing also saw innovations beyond simple text chunking. Jina AI's Late Chunking [24] postponed chunking until after embedding, preserving more contextual information. However, this method requires specific embedding model architectures. dsRAG [25] introduced "automatic context," leveraging LLMs to enrich text chunks with contextual information for better retrieval. Anthropic's Contextual Retrieval [26] also incorporated "Contextual Chunking" using LLMs to generate explanations for each chunk. Meta-Chunking [45] focused on identifying linguistically coherent sentence groups within paragraphs. Multi-granularity hybrid chunking [46] dynamically adjusted chunk size based on query needs. While diverse, these approaches highlight the increasing value of adding contextual information to text chunks, often using LLMs to bridge the semantic gap. RAGFlow also incorporated LLM-powered information extraction for text chunks to improve recall.

**Hybrid Search: The Power of Combination**

A pivotal moment was the publication of IBM Research's BlendedRAG technical report [7] in April 2024. It empirically demonstrated that combining multiple retrieval strategies – vector search, sparse vector search, and full-text search – significantly improved RAG performance. This makes intuitive sense: vector search captures semantic similarity, while full-text and sparse vector search excel at precise matching.

Hybrid search typically relies on specialized databases. However, implementing a truly effective full-text search is non-trivial. Sparse vectors, while aiming to replace full-text search, struggle with specialized terminology not present in their training data. A robust full-text search needs to handle phrase queries and performance efficiently.

RAGFlow was an early adopter of hybrid search, utilizing Elasticsearch as its backend. Queries in RAGFlow undergo sophisticated analysis, including stop word removal, term weighting, and the generation of bi-gram phrase queries. A simple question like "When should teachers ask questions in the original text?" could translate into a complex Elasticsearch query incorporating multiple phrases and weights. This level of sophistication requires inverted indexes with positional information. Furthermore, the default "OR" behavior for keywords in full-text search necessitates dynamic query pruning techniques to maintain performance. Beyond Elasticsearch, the Infinity database also offers this level of sophisticated full-text search capability.

Our experiments with the Infinity database on a public benchmark dataset clearly demonstrate the benefits of three-way hybrid retrieval (vector, sparse vector, and full-text search) compared to single or dual retrieval methods. The results, shown in the image you provided, highlight the superior ranking quality achieved with a multi-pronged approach, validating the findings of BlendedRAG. The far right of the chart also demonstrates the additional benefit of tensor-based re-ranking, a topic we'll discuss later.

OpenAI's acquisition of Rockset in June 2024, following earlier integrations with Qdrant in GPT-4 Turbo, suggests a strategic shift towards incorporating robust data infrastructure to power RAG-based services. Rockset's capabilities as a potential Elasticsearch alternative, coupled with its cloud-native architecture, make it a valuable asset for OpenAI's SaaS offerings.

**Ranking Models: Refining the Search Results**

Ranking lies at the heart of any search system. In the context of RAG, ranking involves two key components: embedding models for initial coarse-grained retrieval and reranker models for fine-tuning the results. Training of these models often shares common methodologies. Embedding models, typically using an encoder architecture, are trained to bring semantically similar text closer in vector space. Rerankers, often employing a cross-encoder architecture, are trained to predict the relevance score between a query and a document.

As illustrated in the image you provided, embedding models encode queries and documents separately, losing token-level interaction information, making them suitable for initial filtering. Cross-encoders, on the other hand, process the query and document together, capturing fine-grained interactions and achieving higher ranking accuracy. However, this comes at a computational cost, limiting their use to reranking a smaller set of top-k results from the initial retrieval.

The MTEB benchmark provides valuable insights into the performance of embedding and reranker models. While cross-encoders dominated the reranking leaderboard in the first half of 2024, the latter half saw a rise in LLM-based rerankers, such as gte-Qwen2-7B [31]. These models, often based on decoder architectures like Qwen2 7B, offer superior performance but at an even higher computational cost.

To balance performance and cost, late-interaction models based on tensor representations gained traction. This approach, illustrated in the image you provided, involves storing token-level embeddings during indexing. At query time, the similarity between all token pairs between the query and document is calculated and aggregated to produce a relevance score. This method captures token interactions similar to cross-encoders but with significantly lower computational overhead, allowing for ranking within the database itself. This capability allows for reranking a larger pool of initial results, potentially recovering relevant documents missed in the initial coarse-grained retrieval. Our experiments with the Infinity database demonstrate the significant improvement in ranking quality achieved by adding tensor reranking to single, dual, and triple retrieval strategies.

Tensor-based reranking, initially proposed in the ColBERT papers [32, 33] in 2020, gained significant attention in early 2024. However, its practical implementation was initially limited by the lack of suitable database support, often relying on Python libraries like RAGatouille [34]. Vespa [35] was an early adopter of tensor-based ranking, and we integrated it into the Infinity database mid-year. While still relatively nascent, the ecosystem around tensor-based reranking is rapidly evolving, with models like JaColBERT [36] for Japanese and Jina's jina-colbert-v2 [37] for multilingual text emerging. As more models become available, tensor-based reranking is poised for widespread adoption in 2025, particularly in multi-modal RAG scenarios, as discussed below.

**Bridging the Semantic Gap:  Understanding User Intent**

The open-sourcing of Microsoft's GraphRAG [8] in the first half of 2024 generated immense excitement, quickly garnering tens of thousands of GitHub stars. This enthusiasm stemmed from its potential to address the persistent challenge of the semantic gap in RAG.

Various approaches emerged to tackle this issue. RAPTOR [9] pre-clusters text and uses LLMs to generate summaries, which are then indexed alongside the original text. This provides a more macroscopic view of the content, aiding in answering ambiguous or multi-hop questions. RAGFlow integrated RAPTOR to enhance its ability to handle complex queries. SiReRAG [17], building upon RAPTOR, introduced a more granular approach to retrieval, considering both semantic similarity and relevance. It extracts named entities from text chunks and constructs a hierarchical tree structure based on their relationships, offering multiple levels of granularity for retrieval.

SiReRAG shares conceptual similarities with GraphRAG. GraphRAG leverages LLMs to automatically extract named entities and build knowledge graphs. These graphs, including entities, relationships, and community summaries (clusters of related entities), are used alongside the original documents for hybrid retrieval. This interconnected information provides crucial context for addressing complex queries. Several other GraphRAG implementations emerged concurrently, such as Ant Group's KAG [10] and Nebula Graph's GraphRAG [11], each with slightly different focuses. KAG emphasizes the integrity and explainability of the knowledge graph, while Nebula Graph's solution prioritizes integration with popular LLM frameworks like LangChain and LlamaIndex and deep integration with their Nebula Graph database.

A significant challenge with GraphRAG is its high token consumption. This led to the development of lighter-weight variations like Fast GraphRAG [12], LightRAG [13], and Microsoft's upcoming LazyGraphRAG [14]. Fast GraphRAG skips the community summarization step to reduce LLM calls, while LightRAG eliminates communities altogether. LazyGraphRAG takes a more radical approach, forgoing LLMs for knowledge graph extraction and relying on smaller, local models for named entity recognition and co-occurrence-based community detection.

Another approach to reduce GraphRAG's overhead is to fine-tune smaller, specialized models for knowledge graph extraction, as demonstrated by Triplex [16], which utilizes the 3B parameter Phi-3 model.

The core idea behind GraphRAG is to enrich the original documents with structured information extracted by LLMs, organized in a graph format to facilitate connections between different parts of the document. The value of the knowledge graph lies not in manual inspection but in providing additional context and evidence for complex queries. Despite the limitations of LLM-extracted knowledge graphs (including potential noise), subsequent research has focused on organizing entities more effectively. KG-Retriever [20] creates a multi-layered graph index combining the knowledge graph and raw data, while Mixture-of-PageRanks [21] incorporates time-based relevance into personalized PageRank. This area is expected to see continued innovation in 2025. The concept of "hippocampal indexing," inspired by how the human brain recalls information, as explored in HippoRAG [15], also gained traction, suggesting new ways to leverage knowledge graphs for retrieval. The image you provided effectively illustrates the core concepts of these graph-based approaches, potentially including elements of Graph Neural Networks (GNNs), which were also explored in the context of knowledge graph-based question answering ([18], [19]). However, GNN-based approaches often require task-specific training data, increasing customization costs and falling outside the primary scope of this overview.

From an engineering perspective, graph databases are a natural fit for implementing GraphRAG, as seen in KAG and Nebula Graph. RAGFlow, however, adopted a different approach, leveraging the capabilities of search engines. In RAGFlow, entities and relationships within the knowledge graph are textual descriptions. A well-featured full-text index allows filtering based on keywords within the "source entity name" and "target entity name" fields, enabling efficient subgraph retrieval. Furthermore, if the database offers seamless integration of full-text and vector indexes, it can provide convenient hybrid search for GraphRAG, incorporating the textual descriptions of entities, relationships, and communities alongside vector embeddings. With the addition of a "type" field, this data can be stored alongside the original text chunks, forming the basis of a HybridRAG [22] approach. This highlights the potential of feature-rich databases to simplify the implementation of sophisticated RAG techniques, even those involving complex graph structures. This is a key motivation behind our development of the RAG-specific database, Infinity.

**Agentic RAG and Memory Management**

"Agentic" was a buzzword in the RAG space in 2024, with many declaring it the "Year of the Agent."  Regardless of the label, agents significantly impacted the LLM ecosystem, particularly in their symbiotic relationship with RAG. RAG acts as a crucial tool for agents, providing access to internal data. Conversely, agents enhance RAG capabilities, leading to "Agentic RAG" approaches like Self-RAG [39] and Adaptive RAG. These advanced RAG systems, as depicted in the image you provided, enable more complex and adaptable information retrieval workflows. Implementing Agentic RAG requires agent frameworks with "closed-loop" capabilities, often referred to as "reflection" in Andrew Ng's agent design patterns. LangGraph [40] was an early framework to implement this, and RAGFlow introduced similar functionality mid-year.

A defining characteristic of agents in 2024 was the widespread adoption of workflows, enabling integration with various systems and controlled execution. However, agents encompass more than just workflows, including reasoning and decision-making. The latter half of 2024 saw increased activity in this area. Integrating RAG with these reasoning-capable agents unlocks new possibilities. For example, systems with multiple autonomous agents (as illustrated in the image you provided [41]) can decompose complex queries into subtasks, with each agent responsible for specific functionalities, improving efficiency and accuracy. Examples include Detector Agents for identifying potential flaws in queries, Thought Agents for synthesizing retrieved information and generating reasoning steps, and Answer Agents for producing final answers based on the Thought Agent's output.

RARE [42] further exemplifies this trend, augmenting RAG with the Monte Carlo Tree Search (MCTS) framework to enhance reasoning capabilities through iterative query generation and refinement.

This increasing interplay between RAG and agents necessitates that RAG systems provide memory management capabilities beyond document retrieval. This includes storing user conversation history, personalized information, and other contextual data. Agents need to access this real-time context in addition to querying document knowledge. The rapid popularity of the open-source project Mem0, which defines APIs for memory management, underscores this need. However, while the core functionalities of memory management (real-time filtering and search) are relatively straightforward, the real challenge lies in integrating memory with reasoning capabilities to unlock more sophisticated enterprise-level applications. Implementing this within a standardized RAG engine is a logical, cost-effective, and user-friendly approach and is expected to be a major focus in 2025.

The question remains: will RAG evolve into an agent platform, or will agent platforms enhance their RAG capabilities? The answer is uncertain. Just as in the digital era, the lines blurred between data warehousing and business-focused platform development, both paths are possible. In the age of LLMs, RAG could be seen as the equivalent of the traditional database, while agents, with reduced customization needs, could become standardized application-layer products. The future likely involves a dynamic interplay between technological depth and rapid product iteration, with closer integration between different parts of the software ecosystem. LangGraph's late-year release of an LLM-based agent interoperability protocol hints at this future, where agents can interact and form interconnected ecosystems.

**Multi-Modal RAG:  Expanding the Boundaries**

Multi-modal RAG is another area poised for rapid growth in 2025, driven by advancements in key underlying technologies in 2024.

The emergence of powerful Vision Language Models (VLMs), as illustrated in the image you provided, marked a significant step forward. VLMs evolved beyond simple image recognition to achieve a deeper understanding of visual content, including complex enterprise documents. Even smaller models like the open-source PaliGemma [43] demonstrate this capability.

This advancement paves the way for true multi-modal RAG. Imagine being able to ask a question and have the RAG system retrieve not just relevant text but also the specific images and charts within a collection of PDFs that contain the answer. The VLM can then be used to generate the final answer, leveraging both visual and textual information.

One approach to achieving this builds on previous advancements: using models to convert multi-modal documents into text, indexing the text, and performing retrieval. Another, more direct approach, leverages the capabilities of VLMs to generate embeddings directly from images, bypassing the complex OCR process. ColPali [44], introduced in the summer of 2024, pioneered this approach by treating an image as a sequence of patches and generating embeddings for each patch, resulting in a tensor representation for the entire image. Ranking is then performed using these tensor representations.

The entire retrieval process, as shown in the image you provided, requires databases that not only support tensor-based reranking but also multi-vector indexing during the vector search stage – capabilities already present in our Infinity database.

The image you provided also showcases the performance of various multi-modal RAG systems, highlighting the strong performance of tensor-based late-interaction models.

The question arises: should multi-modal RAG rely on direct embedding generation or first convert documents to text using generalized OCR?  While ColPali's initial work advocated for bypassing OCR, their comparisons were against older CNN-based OCR models (our "Generation 1"). Currently, both approaches are viable, leveraging the increasing capabilities of multi-modal models. If embedding is considered a "general-purpose" solution, then Encoder-Decoder based OCR can be seen as a more "specialized" approach, as certain document types may still benefit from task-specific training. RAG, at its core, prioritizes practical implementation, so specific solutions may be optimal for specific tasks in the short term, with a potential convergence towards unified approaches in the future.

The stage is set for rapid growth and evolution in multi-modal RAG in 2025, and we are committed to incorporating these capabilities into RAGFlow at the appropriate time.

### Conclusion:  Looking Ahead

2024 has undeniably been a transformative year for RAG. It has solidified its position as not just a feature but an architectural pattern – a complex ecosystem that goes far beyond simple keyword search. RAG can be viewed as the evolution of enterprise search in the age of large language models. While it may not attract the same level of funding hype as LLMs themselves, its practical indispensability and inherent complexity are undeniable.

Reference in https://zhuanlan.zhihu.com/p/14116449727