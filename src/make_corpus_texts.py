import os, random, textwrap

OUT = "data/corpus_txt"
os.makedirs(OUT, exist_ok=True)

random.seed(42)

paragraphs = [
    "Artificial intelligence is transforming the way organizations analyze data and make decisions. "
    "Machine learning models are now capable of predicting outcomes with remarkable accuracy.",

    "In this report, we evaluate the performance of several algorithms using multiple evaluation metrics. "
    "Our focus is on precision, recall, and the F1-score, which help determine the robustness of each model.",

    "The dataset used in this experiment was collected from open-source repositories. "
    "It contains thousands of records, each with multiple numerical and categorical features.",

    "According to the latest research in cognitive computing, human–computer interaction can be improved "
    "by integrating emotional intelligence and natural language understanding.",

    "Meeting minutes recorded on Tuesday highlight the need for better coordination among project teams. "
    "The data collection phase was delayed due to unexpected hardware failures.",

    "Our financial analysis shows an upward trend in quarterly growth despite minor fluctuations in expenses. "
    "We recommend continued investment in automation tools to sustain productivity.",

    "The experimental design followed a randomized control structure. "
    "All variables were monitored under consistent environmental conditions to ensure reproducibility.",

    "Cybersecurity remains a major concern in cloud infrastructures. "
    "Proper encryption and access control mechanisms are essential to safeguard sensitive information.",

    "User feedback from the pilot program indicates positive engagement with the system’s adaptive interface. "
    "Future updates will focus on improving load times and accessibility compliance.",

    "Environmental studies emphasize the role of sustainable resource management. "
    "Reducing carbon emissions remains a top priority for both industrial and research sectors."
]

N = 10000 
for i in range(N):
    doc_paragraphs = random.sample(paragraphs, k=random.randint(4, 8))
    text = "\n\n".join(doc_paragraphs)
    wrapped = "\n".join(textwrap.wrap(text, width=95))

    with open(f"{OUT}/doc_{i:05d}.txt", "w", encoding="utf-8") as f:
        f.write(wrapped)

print(f"Wrote {N} meaningful text files to {OUT}")
