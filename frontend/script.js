const API_URL = "http://127.0.0.1:8000/predict"; // Change this after deployment

const emojiMap = {
  positive: "😊",
  neutral: "😐",
  negative: "😞",
};

async function predictSentiment() {
  const text = document.getElementById("textInput").value;
  const loader = document.getElementById("loader");
  const resultCard = document.getElementById("resultCard");

  if (!text.trim()) {
    alert("Please enter some text first.");
    return;
  }

  resultCard.classList.add("hidden");
  loader.classList.remove("hidden");

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text }),
    });

    const data = await response.json();

    document.getElementById("emoji").textContent = emojiMap[data.prediction] || "🤔";
    document.getElementById("predictionText").textContent = data.prediction;
    document.getElementById("confidenceText").textContent = data.confidence
      ? `Confidence: ${data.confidence}%`
      : "";

    loader.classList.add("hidden");
    resultCard.classList.remove("hidden");
  } catch (error) {
    loader.classList.add("hidden");
    alert("Error connecting to the API. Make sure the backend server is running.");
    console.error(error);
  }
}
