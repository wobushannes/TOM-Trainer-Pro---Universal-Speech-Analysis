
TOM TRAINER PRO - UNIVERSAL SPEECH ANALYSIS

Enterprise-Grade LoRA Training & Theory of Mind Analysis Platform

Autor:    Johannes Wobus
Lizenz:   MIT
Stand:    Juli 2026

https://www.youtube.com/watch?v=CNT9nQdYa4E
https://www.youtube.com/watch?v=JR9oRLHoD3w
https://www.youtube.com/watch?v=lSDSjdFZsVo


ÜBERBLICK
================================================================================

TOM Trainer Pro ist eine Plattform für Theory of Mind (ToM) Analyse und
LoRA-basiertes Fine-Tuning von Large Language Models (LLMs). Das System
erkennt verborgene emotionale Muster, Täuschungsindikatoren und kommunikative
Strategien in menschlicher Sprache.

KERN-FUNKTIONEN
================================================================================

  Training         LoRA-Fine-Tuning mit Echtzeit-Monitoring
                   (Loss, GPU-Auslastung, Konvergenz)

  Reports          Automatisierte HTML-Trainingsberichte mit
                   Qualitätsmetriken

  Dokumenten-      Chunk-basierte ToM-Analyse von PDFs und
  Analyse          Textdokumenten

  Echtzeit-        Satzweise Analyse mit detaillierter
  Analyse          Interpretation

  Data Quality     Metriken zu Datenqualität, Emotions-Diversität
                   und Domain-Balance

UNTERSTÜTZTE MODELLE
================================================================================

  - Qwen2.5-Familie: 1.5B, 4B, 7B Instruct
  - Mistral AI: Mistral-7B-Instruct-v0.3
  - Microsoft: DialoGPT-medium (Legacy)


SYSTEMVORAUSSETZUNGEN
================================================================================

  Python:       3.9 oder höher
  RAM:          16 GB (32 GB empfohlen)
  GPU:          NVIDIA mit 8 GB+ VRAM (zwingend für Training)
  Speicher:     20 GB+ für Modelle und Daten



REQUIREMENTS.TXT
================================================================================

  torch>=2.0.0
  transformers>=4.36.0
  peft>=0.7.0
  datasets>=2.14.0
  accelerate>=0.25.0
  bitsandbytes>=0.41.0
  scikit-learn>=1.3.0
  matplotlib>=3.7.0
  pandas>=2.0.0
  numpy>=1.24.0
  textstat>=0.7.0
  pymupdf>=1.23.0
  tkinter  # Standard in Python enthalten


TRAININGSDATEN-FORMAT
================================================================================

Das System erwartet JSON- oder JSONL-Dateien im folgenden Format:

{
  "input": "Der zu analysierende Text",
  "tom": {
    "deception_likelihood": 0.75,
    "detected_true_emotions": ["Trauer", "Wut"],
    "hidden_intent_candidate": "Verschleierung von Schuld"
  },
  "emotion_dynamics": {
    "emotional_volatility": 0.82,
    "micro_expression_hypothesis": "Unterdrückter Ärger"
  },
  "communication_style": {
    "communication_style": "Vermeidend",
    "persuasion_strategy": "Emotionale Manipulation"
  }
}

FELDBESCHREIBUNG

  Feld                                   Typ        Bereich    Beschreibung
  --------------------------------------------------------------------------
  input                                  String     -          Der zu analysierende Text
  tom.deception_likelihood               Float      0.0-1.0    Täuschungswahrscheinlichkeit
  tom.detected_true_emotions             Array      -          Erkannte Emotionen
  tom.hidden_intent_candidate            String     -          Vermutete Absicht
  emotion_dynamics.emotional_volatility  Float      0.0-1.0    Emotionale Volatilität
  emotion_dynamics.micro_expression_     String     -          Hypothese zu Mikroausdrücken
    hypothesis
  communication_style.communication_style String    -          Kommunikationsstil
  communication_style.persuasion_strategy String     -          Überzeugungsstrategie


SCHNELLSTART MIT DEMO-DATENSATZ
================================================================================

Ein Demo-Datensatz mit 10 Einträgen ist in "demo_training_data.json" enthalten.

  1. GUI starten:                python tom_trainer_pro_gui.py
  2. Modell auswählen:           z.B. Qwen/Qwen2.5-1.5B-Instruct
  3. "Durchsuchen"               → demo_training_data.json auswählen
  4. "Rohdaten zu Training konvertieren" klicken
  5. "Training starten"          (3-5 Episoden empfohlen)
  6. Nach Training:              "Trainiertes LoRA laden" für Analysen


WICHTIGER HINWEIS ZU TRAININGSDATEN
================================================================================

  Ich biete KEINEN Support für die Generierung von Trainingsdaten oder die
  Annotation von ToM-Merkmalen.

  Die Erstellung hochwertiger Trainingsdaten erfordert:
    - Tiefgehende psycholinguistische Expertise
    - Verständnis von Theory of Mind Konzepten
    - Erfahrung mit Emotionsannotation
    - Kenntnis von Kommunikationsstrategien


KONTAKT FÜR PROFESSIONELLE ZUSAMMENARBEIT
================================================================================

Ich habe mehrere LoRAs mit diesen Formaten trainiert und biete gegen Honorar:

  - Datenannotation:            Professionelle ToM-Annotation
  - LoRA-Training:              Custom LoRAs für spezifische Fälle
  - Qualitätssicherung:         Data Quality Audits & Optimierung
  - Deployment:                 Integration in bestehende Systeme


KONTAKTAUFNAHME (BEWERBUNGSVERFAHREN)
================================================================================

Bevor du mich kontaktierst, sende BITTE folgende Informationen:

  Projekt-Art:          [Forschung / Unternehmen / Startup / Anderes]
  Anwendungsbereich:    [z.B. Politische Analyse / Marktforschung]
  Datenvolumen:         [Anzahl der Sätze / Dokumente]
  Zeitrahmen:           [Deadline / Projektlaufzeit]
  Budget:               [Rahmen / Indikation]
  Vorwissen:            [Erfahrung mit NLP / ToM / LoRA]
  Erwartungen:          [Was soll das System leisten?]

  E-Mail:               blende_32@protonmail.com
  Betreff:              [ToM-Projekt] - [Ihre Organisation]
  Antwortzeit:          48-72 Stunden (bei vollständigen Angaben)

  Ich antworte NUR auf Anfragen, die ALLE oben genannten Punkte enthalten.
  Unvollständige Anfragen werden nicht bearbeitet.



USE CASES
================================================================================

  1. Politische Redeanalyse
     Beispiel:
     text = "Die Opposition verbreitet bewusst Falschinformationen..."
     result = analyze_text(text)
     # → Täuschung: 65%, Emotionen: ['Verärgerung', 'Verachtung']

  2. Forensische Linguistik
     - Täuschung in Zeugenaussagen
     - Analyse von Verhörprotokollen
     - Bewertung von Glaubwürdigkeit

  3. Psychologische Forschung
     - Emotionsdynamik in Therapiegesprächen
     - Kommunikationsstile in der Paartherapie
     - ToM-Entwicklung bei Kindern

  4. Sicherheitsanalyse
     - Früherkennung von Manipulationsversuchen
     - Whistleblower-Analyse
     - Risikobewertung in Verhandlungen


DISCLAIMER
================================================================================

  Dieses Tool ist für Forschungszwecke konzipiert. Die Analyseergebnisse sind
  probabilistisch und nicht als alleinige Entscheidungsgrundlage geeignet.
  Fehlinterpretationen sind möglich. Verwendung auf eigene Verantwortung.


BEITRÄGE (CONTRIBUTIONS)
================================================================================

  Contributions sind willkommen!

  1. Fork das Repository
  2. Feature Branch erstellen: git checkout -b feature/AmazingFeature
  3. Commit:                    git commit -m 'Add some AmazingFeature'
  4. Push:                      git push origin feature/AmazingFeature
  5. Pull Request öffnen


LIZENZ
================================================================================

  MIT License – siehe LICENSE-Datei im Repository.


KONTAKT
================================================================================

  E-Mail:   blende_32@protonmail.com (nur für vollständige Anfragen)

