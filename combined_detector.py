import os
from transformers import pipeline
from voice_extraction import extract_features, assess_parkinsons, record_audio
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np

class ParkinsonDiseaseDetector:
    def __init__(self):
        self.handwriting_model = pipeline("image-classification", 
                                        "gianlab/swin-tiny-patch4-window7-224-finetuned-parkinson-classification")
        self.setup_gui()

    def setup_gui(self):
        self.window = tk.Tk()
        self.window.title("Parkinson's Disease Multi-Modal Detector")
        self.window.geometry("600x400")

        # Voice Input Frame
        voice_frame = tk.LabelFrame(self.window, text="Voice Analysis", padx=10, pady=10)
        voice_frame.pack(padx=10, pady=5, fill="x")

        tk.Button(voice_frame, text="Record Voice (5s)", command=self.record_voice).pack(side=tk.LEFT, padx=5)
        tk.Button(voice_frame, text="Upload Voice File", command=self.upload_voice).pack(side=tk.LEFT, padx=5)
        self.voice_label = tk.Label(voice_frame, text="No voice file selected")
        self.voice_label.pack(side=tk.LEFT, padx=5)

        # Handwriting Input Frame
        handwriting_frame = tk.LabelFrame(self.window, text="Handwriting Analysis", padx=10, pady=10)
        handwriting_frame.pack(padx=10, pady=5, fill="x")

        tk.Button(handwriting_frame, text="Upload Handwriting Image", 
                 command=self.upload_handwriting).pack(side=tk.LEFT, padx=5)
        self.handwriting_label = tk.Label(handwriting_frame, text="No image file selected")
        self.handwriting_label.pack(side=tk.LEFT, padx=5)

        # Results Frame
        results_frame = tk.LabelFrame(self.window, text="Results", padx=10, pady=10)
        results_frame.pack(padx=10, pady=5, fill="x")

        tk.Button(results_frame, text="Analyze", command=self.analyze).pack(pady=5)
        self.result_text = tk.Text(results_frame, height=10, width=60)
        self.result_text.pack(pady=5)

        # Initialize file paths
        self.voice_file = None
        self.handwriting_file = None

    def record_voice(self):
        output_file = "recorded_voice.wav"
        record_audio(duration=5, output_file=output_file)
        self.voice_file = output_file
        self.voice_label.config(text=f"Recorded: {os.path.basename(output_file)}")

    def upload_voice(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if file_path:
            self.voice_file = file_path
            self.voice_label.config(text=f"Selected: {os.path.basename(file_path)}")

    def upload_handwriting(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")])
        if file_path:
            self.handwriting_file = file_path
            self.handwriting_label.config(text=f"Selected: {os.path.basename(file_path)}")

    def analyze_voice(self):
        if not self.voice_file or not os.path.exists(self.voice_file):
            return "No", 0, []
        
        features = extract_features(self.voice_file)
        return assess_parkinsons(features)

    def analyze_handwriting(self):
        if not self.handwriting_file or not os.path.exists(self.handwriting_file):
            return "No", 0
        
        result = self.handwriting_model(self.handwriting_file)
        if result and isinstance(result, list) and len(result) > 0:
            prediction = result[0]
            has_parkinsons = 'parkinson' in prediction['label'].lower()
            confidence = prediction['score']
            return "Yes" if has_parkinsons else "No", confidence
        return "No", 0

    def late_fusion(self, voice_result, handwriting_result):
        """Combine results using late fusion technique"""
        voice_pred, voice_risk, voice_details = voice_result
        hw_pred, hw_confidence = handwriting_result

        # Convert predictions to numerical values
        voice_score = 1 if voice_pred == "Yes" else 0
        hw_score = 1 if hw_pred == "Yes" else 0

        # Weight the predictions (can be adjusted based on reliability of each modality)
        voice_weight = 0.6
        hw_weight = 0.4

        combined_score = (voice_score * voice_weight + hw_score * hw_weight)
        final_prediction = "Yes" if combined_score >= 0.5 else "No"
        confidence = combined_score if final_prediction == "Yes" else (1 - combined_score)

        return final_prediction, confidence, voice_risk, voice_details, hw_confidence

    def analyze(self):
        self.result_text.delete(1.0, tk.END)
        
        if not self.voice_file and not self.handwriting_file:
            messagebox.showerror("Error", "Please provide at least one input (voice or handwriting)")
            return

        # Analyze voice
        voice_result = self.analyze_voice()
        
        # Analyze handwriting
        handwriting_result = self.analyze_handwriting()
        
        # Combine results using late fusion
        final_pred, confidence, voice_risk, voice_details, hw_conf = self.late_fusion(
            voice_result, handwriting_result)

        # Display results
        self.result_text.insert(tk.END, f"Final Prediction: {final_pred}\n")
        self.result_text.insert(tk.END, f"Combined Confidence: {confidence:.2f}\n\n")
        
        self.result_text.insert(tk.END, "Voice Analysis:\n")
        self.result_text.insert(tk.END, f"- Prediction: {voice_result[0]}\n")
        self.result_text.insert(tk.END, f"- Risk Factors: {voice_risk}/4\n")
        for detail in voice_details:
            self.result_text.insert(tk.END, f"  * {detail}\n")
        
        self.result_text.insert(tk.END, "\nHandwriting Analysis:\n")
        self.result_text.insert(tk.END, f"- Prediction: {handwriting_result[0]}\n")
        self.result_text.insert(tk.END, f"- Confidence: {hw_conf:.2f}\n")

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    detector = ParkinsonDiseaseDetector()
    detector.run()