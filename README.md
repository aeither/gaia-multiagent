
## Overview

TechTrio Gaia Multiagent is a cutting-edge Living Knowledge System that leverages the power of decentralized AI to provide users with comprehensive, up-to-date insights on the latest tech products. By combining the expertise of top tech influencers, real-time product information, and advanced AI models, TechTrio offers an unparalleled shopping assistant experience for tech enthusiasts.

### Key Features

* **Multi-Agent System**: Utilizes AI agents modeled after three renowned tech reviewers: MKBHD, Unbox Therapy, and MrWhosTheBoss.
* **Living Knowledge Base**: Employs Upstash Vector DB as an inventory system to continuously update information on gadgets and new devices.
* **Decentralized Infrastructure**: Built on GaiaNet nodes, ensuring a robust and scalable system.
* **Impartial Moderation**: Incorporates a moderator agent to summarize and compare reviews, providing balanced conclusions.

### Technical Stack

* **AI Models**: Utilizes various models including Llama, Phi, and Gemma through GaiaNet nodes.
* **Vector Database**: Employs Upstash Vector DB for efficient storage and retrieval of product information.
* **Embedding**: Leverages the bge-base-en-v1.5 model for text embedding.
* **API Integration**: Seamlessly integrates with OpenAI-compatible APIs provided by GaiaNet.

### How It Works

1. **User Query**: The user submits a question about a tech product they're interested in.
2. **Multi-Agent Processing**: The query is processed by AI agents representing MKBHD, Unbox Therapy, and MrWhosTheBoss, each providing their unique perspective.
3. **Knowledge Retrieval**: The system queries the Upstash Vector DB to retrieve the most up-to-date product information.
4. **Review Generation**: Each AI agent generates a review based on their persona and the available information.
5. **Moderation and Summary**: A moderator agent compiles and summarizes the reviews, highlighting key points of agreement and disagreement.
6. **Final Output**: The user receives a comprehensive summary of the product reviews, along with a balanced conclusion to aid in their purchasing decision.
### Key Advantages

* **Diverse Perspectives**: Gain insights from multiple trusted tech influencers in one place.
* **Up-to-Date Information**: The living knowledge system ensures that product information is always current.
* **Balanced View**: The moderator summary provides an impartial overview of different opinions.
* **Time-Saving**: Quickly access expert opinions without watching multiple video reviews.
* **Personalized Recommendations**: Receive tailored advice based on user preferences and needs.

### Future Development Roadmap

* **Expanded Expertise**: Integrate with more tech influencers and experts to broaden the knowledge base.
* **Product Category Expansion**: Cover a wider range of tech products and categories to cater to diverse user needs.
* **User Feedback Mechanisms**: Implement user feedback systems to improve AI responses and enhance the overall user experience.
* **Price Comparison and Availability Features**: Add features to compare prices and check availability of products to facilitate informed purchasing decisions.

### Instructions

Install project dependencies

```bash
uv add crewai crewai-tools streamlit
```

Run your project

```bash
uv run streamlit run plan.py
```

Run Store Vectors

```bash
cd ai && uv npx tsx upsertVectors.ts
```

Run Query

```bash
cd ai && uv npx tsx queryVectors.ts
```

