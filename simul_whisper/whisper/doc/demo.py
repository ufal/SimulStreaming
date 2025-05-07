import whisper

model = whisper.load_model("../large-v3.pt")
print("tady",flush=True)
result = model.transcribe("../../demo-wav/sample.wav")
print("tady",flush=True)
print(result["text"])
