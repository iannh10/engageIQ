"""Configuration constants for EngageIQ."""

from __future__ import annotations


DOMAINS = [
    "Machine Learning",
    "DevOps/K8s",
    "Trending Open-Source",
    "Developer Tools",
    "Cybersecurity",
    "Frontend (React/Web)",
    "B2B SaaS",
    "Blockchain",
    "Python Data Eng",
    "GameDev (C++)",
    "AI Research",
    "Embedded Systems (C/RTOS)",
    "Cloud APIs",
    "Mobile Dev (iOS/Flutter)",
    "Beginner Coding",
]

SOURCES = ["github", "gh_archive", "reddit", "hacker_news"]

DOMAIN_KEYWORDS = {
    "Machine Learning": ["machine learning", "nlp", "classification", "pandas", "model", "feature store"],
    "DevOps/K8s": ["kubernetes", "terraform", "ci/cd", "observability", "helm", "cloud native"],
    "Trending Open-Source": ["open source", "stars", "maintainers", "roadmap", "community", "release"],
    "Developer Tools": ["developer tools", "cli", "sdk", "api", "workflow", "productivity"],
    "Cybersecurity": ["security", "vulnerability", "threat", "auth", "zero trust", "incident"],
    "Frontend (React/Web)": ["react", "web", "typescript", "ui", "next.js", "accessibility"],
    "B2B SaaS": ["saas", "crm", "sales", "retention", "onboarding", "customer success"],
    "Blockchain": ["blockchain", "smart contract", "wallet", "ethereum", "defi", "web3"],
    "Python Data Eng": ["python", "etl", "data pipeline", "airflow", "spark", "warehouse"],
    "GameDev (C++)": ["c++", "unreal", "game engine", "graphics", "physics", "rendering"],
    "AI Research": ["llm", "paper", "benchmark", "alignment", "agents", "evaluation"],
    "Embedded Systems (C/RTOS)": ["embedded", "rtos", "firmware", "c", "microcontroller", "iot"],
    "Cloud APIs": ["cloud api", "serverless", "aws", "gcp", "azure", "integration"],
    "Mobile Dev (iOS/Flutter)": ["ios", "flutter", "swift", "mobile", "android", "app store"],
    "Beginner Coding": ["beginner", "good first issue", "tutorial", "first contribution", "python basics"],
}

PERSONAS = {
    "Sofia": {
        "name": "Sofia ML Student",
        "background": "MSBA student building a visible open-source portfolio before job hunting.",
        "interests": "machine learning, NLP, data pipelines, Python, pandas, beginner-friendly repos",
        "goal": "Find beginner-friendly GitHub repos and ML discussions for visible contribution.",
        "platforms": ["github", "reddit"],
        "time_budget": 5,
        "avoid": "C++, Rust, advanced systems internals",
    },
    "Emma": {
        "name": "Emma Career Switcher",
        "background": "Beginner software learner moving into tech and building a practical public portfolio.",
        "interests": "beginner Python, web development, tutorials, documentation, first contribution, portfolio projects",
        "goal": "Find approachable GitHub issues and learning-friendly discussions that build confidence and visible experience.",
        "platforms": ["github", "reddit", "hacker_news"],
        "time_budget": 4,
        "avoid": "advanced C++ systems, Kubernetes internals, cryptography research, low-level firmware",
    },
    "David": {
        "name": "David DevOps Engineer",
        "background": "Mid-career DevOps engineer establishing cloud-native thought leadership.",
        "interests": "Kubernetes, Terraform, CI/CD, observability, infrastructure, cloud native",
        "goal": "Find high-signal infra repos and discussions where expert commentary adds value.",
        "platforms": ["github", "reddit", "hacker_news"],
        "time_budget": 3,
        "avoid": "general frontend, beginner web dev",
    },
    "Lina": {
        "name": "Lina Data Journalist",
        "background": "Tech journalist tracking open-source and community trend leads.",
        "interests": "trending repos, viral discussions, emerging tools, community drama, growth velocity",
        "goal": "Surface fast-growing opportunities before they go mainstream.",
        "platforms": ["github", "gh_archive", "reddit", "hacker_news"],
        "time_budget": 10,
        "avoid": "",
    },
    "Raj": {
        "name": "Raj Startup Founder",
        "background": "Technical founder growing awareness for a developer tools startup.",
        "interests": "developer productivity, APIs, CLI tools, open-source business models, startup marketing",
        "goal": "Find relevant threads, posts, and repos for authentic product-aware engagement.",
        "platforms": ["reddit", "github", "hacker_news"],
        "time_budget": 4,
        "avoid": "general programming with no developer-tool angle",
    },
}
