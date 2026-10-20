"""Plain-language, patient-facing summary of a doctor-signed report.

Three rules shape this module:

1. **Built only from what the doctor signed.** The input is the doctor's own
   choices -- which findings to tell the patient about, how urgent follow-up
   is, after how many weeks -- never the raw model output. A patient never
   sees an unreviewed probability through this path.
2. **Templates, not generated text.** Every sentence a patient can receive is
   written out below and can be read and checked in advance, in each language.
   A generative model translating a medical report on the fly cannot offer that.
3. **The doctor's free-text note is shown as written**, labelled as theirs and
   untranslated, rather than machine-translated.

The Hindi and Marathi wording was written for this prototype and has not been
reviewed by a clinician who is a native speaker. Have it checked before any
real patient reads it.
"""

from __future__ import annotations

from typing import Any

LANGUAGES = ("en", "hi", "mr")

# label -> language -> (plain name, one-sentence explanation)
FINDINGS: dict[str, dict[str, tuple[str, str]]] = {
    "No Finding": {
        "en": ("No abnormality seen", "Nothing unusual was found on this X-ray."),
        "hi": ("कोई असामान्यता नहीं", "इस एक्स-रे में कुछ भी असामान्य नहीं मिला।"),
        "mr": ("कोणतीही असामान्यता नाही", "या एक्स-रेमध्ये काहीही असामान्य आढळले नाही."),
    },
    "Enlarged Cardiomediastinum": {
        "en": (
            "Widened centre of the chest",
            "The area in the middle of your chest, around the heart and large blood "
            "vessels, looks wider than usual.",
        ),
        "hi": (
            "छाती का बीच का हिस्सा चौड़ा",
            "दिल और बड़ी रक्त नलियों के आसपास छाती का बीच का हिस्सा सामान्य से अधिक चौड़ा दिखता है।",
        ),
        "mr": (
            "छातीचा मधला भाग रुंद",
            "हृदय आणि मोठ्या रक्तवाहिन्यांच्या भोवतीचा छातीचा मधला भाग नेहमीपेक्षा रुंद दिसतो.",
        ),
    },
    "Cardiomegaly": {
        "en": ("Enlarged heart", "Your heart looks larger than usual on the X-ray."),
        "hi": ("दिल का आकार बढ़ा हुआ", "एक्स-रे में आपका दिल सामान्य से बड़ा दिखता है।"),
        "mr": ("हृदयाचा आकार वाढलेला", "एक्स-रेमध्ये तुमचे हृदय नेहमीपेक्षा मोठे दिसते."),
    },
    "Lung Opacity": {
        "en": (
            "Hazy area in the lung",
            "Part of a lung looks cloudier than normal. This can have many causes, "
            "such as infection or fluid.",
        ),
        "hi": (
            "फेफड़े में धुंधला हिस्सा",
            "फेफड़े का एक हिस्सा सामान्य से अधिक धुंधला दिखता है। इसके कई कारण हो सकते हैं, जैसे संक्रमण या पानी।",
        ),
        "mr": (
            "फुफ्फुसात धूसर भाग",
            "फुफ्फुसाचा एक भाग नेहमीपेक्षा धूसर दिसतो. याची अनेक कारणे असू शकतात, जसे संसर्ग किंवा पाणी.",
        ),
    },
    "Lung Lesion": {
        "en": ("Spot in the lung", "There is a small area in a lung that needs a closer look."),
        "hi": ("फेफड़े में एक धब्बा", "फेफड़े में एक छोटा हिस्सा है जिसकी और जाँच की ज़रूरत है।"),
        "mr": ("फुफ्फुसात एक डाग", "फुफ्फुसात एक लहान भाग आहे ज्याची अधिक तपासणी आवश्यक आहे."),
    },
    "Edema": {
        "en": (
            "Fluid in the lungs",
            "There may be extra fluid in the lungs, which is often related to how the "
            "heart is working.",
        ),
        "hi": (
            "फेफड़ों में पानी",
            "फेफड़ों में अतिरिक्त पानी हो सकता है, जो अक्सर दिल के काम से जुड़ा होता है।",
        ),
        "mr": (
            "फुफ्फुसात पाणी",
            "फुफ्फुसांमध्ये जास्तीचे पाणी असू शकते, जे बहुतेक वेळा हृदयाच्या कार्याशी संबंधित असते.",
        ),
    },
    "Consolidation": {
        "en": (
            "Dense area in the lung",
            "Part of a lung looks filled in, which is often caused by infection.",
        ),
        "hi": (
            "फेफड़े में घना हिस्सा",
            "फेफड़े का एक हिस्सा भरा हुआ दिखता है, जो अक्सर संक्रमण के कारण होता है।",
        ),
        "mr": (
            "फुफ्फुसात दाट भाग",
            "फुफ्फुसाचा एक भाग भरलेला दिसतो, जे बहुतेक वेळा संसर्गामुळे होते.",
        ),
    },
    "Pneumonia": {
        "en": (
            "Possible chest infection (pneumonia)",
            "The X-ray shows signs that can be caused by a lung infection.",
        ),
        "hi": (
            "संभावित छाती का संक्रमण (निमोनिया)",
            "एक्स-रे में ऐसे संकेत हैं जो फेफड़ों के संक्रमण से हो सकते हैं।",
        ),
        "mr": (
            "छातीचा संभाव्य संसर्ग (न्यूमोनिया)",
            "एक्स-रेमध्ये अशी चिन्हे आहेत जी फुफ्फुसाच्या संसर्गामुळे असू शकतात.",
        ),
    },
    "Atelectasis": {
        "en": (
            "Partly collapsed lung area",
            "A small part of a lung is not fully inflated. This is common and often mild.",
        ),
        "hi": (
            "फेफड़े का हिस्सा आंशिक रूप से सिकुड़ा",
            "फेफड़े का एक छोटा हिस्सा पूरी तरह नहीं फूला है। यह आम है और अक्सर हल्का होता है।",
        ),
        "mr": (
            "फुफ्फुसाचा भाग अंशतः आकुंचित",
            "फुफ्फुसाचा एक लहान भाग पूर्ण फुगलेला नाही. हे सामान्य आहे आणि बहुतेक वेळा सौम्य असते.",
        ),
    },
    "Pneumothorax": {
        "en": (
            "Air around the lung",
            "There may be air between the lung and the chest wall, which can stop the "
            "lung from expanding fully.",
        ),
        "hi": (
            "फेफड़े के आसपास हवा",
            "फेफड़े और छाती की दीवार के बीच हवा हो सकती है, जिससे फेफड़ा पूरी तरह नहीं फैल पाता।",
        ),
        "mr": (
            "फुफ्फुसाभोवती हवा",
            "फुफ्फुस आणि छातीच्या भिंतीमध्ये हवा असू शकते, ज्यामुळे फुफ्फुस पूर्णपणे पसरू शकत नाही.",
        ),
    },
    "Pleural Effusion": {
        "en": (
            "Fluid around the lung",
            "There may be fluid collected between the lung and the chest wall.",
        ),
        "hi": ("फेफड़े के आसपास पानी", "फेफड़े और छाती की दीवार के बीच पानी जमा हो सकता है।"),
        "mr": ("फुफ्फुसाभोवती पाणी", "फुफ्फुस आणि छातीच्या भिंतीमध्ये पाणी जमा झालेले असू शकते."),
    },
    "Pleural Other": {
        "en": (
            "Change in the lining of the lung",
            "The thin lining around the lung looks thickened or changed.",
        ),
        "hi": ("फेफड़े की परत में बदलाव", "फेफड़े के चारों ओर की पतली परत मोटी या बदली हुई दिखती है।"),
        "mr": ("फुफ्फुसाच्या आवरणात बदल", "फुफ्फुसाभोवतीचे पातळ आवरण जाड किंवा बदललेले दिसते."),
    },
    "Fracture": {
        "en": (
            "Possible broken bone",
            "One of the bones seen on the X-ray, such as a rib, may be broken.",
        ),
        "hi": ("संभावित हड्डी टूटना", "एक्स-रे में दिख रही कोई हड्डी, जैसे पसली, टूटी हो सकती है।"),
        "mr": ("हाड मोडण्याची शक्यता", "एक्स-रेमध्ये दिसणारे एखादे हाड, जसे बरगडी, मोडलेले असू शकते."),
    },
    "Support Devices": {
        "en": (
            "Medical tube or device",
            "A medical device such as a tube, line or pacemaker is visible. This is "
            "expected if you have one.",
        ),
        "hi": (
            "चिकित्सा नली या उपकरण",
            "कोई चिकित्सा उपकरण जैसे नली, लाइन या पेसमेकर दिख रहा है। अगर आपके पास ऐसा उपकरण है "
            "तो यह अपेक्षित है।",
        ),
        "mr": (
            "वैद्यकीय नळी किंवा उपकरण",
            "नळी, लाइन किंवा पेसमेकरसारखे वैद्यकीय उपकरण दिसत आहे. तुमच्याकडे असे उपकरण असल्यास हे अपेक्षित आहे.",
        ),
    },
}

TEXT: dict[str, dict[str, str]] = {
    "en": {
        "title": "Your chest X-ray summary",
        "none": "Your doctor has reviewed your chest X-ray. No further action is needed "
        "for this X-ray.",
        "routine": "Your doctor has reviewed your chest X-ray and recommends a routine follow-up.",
        "soon": "Your doctor has reviewed your chest X-ray and would like to see you again soon.",
        "urgent": "Your doctor has reviewed your chest X-ray and needs to talk to you urgently. "
        "Please contact your doctor or hospital today.",
        "step_weeks": "Please get a follow-up check in about {weeks} weeks.",
        "step_none": "Keep this summary for your records.",
        "step_routine": "Have routine check-ups as your doctor advises.",
        "step_soon": "Please book an appointment with your doctor soon.",
        "step_urgent": "Contact your doctor today. If you have severe breathlessness or chest "
        "pain, go to the nearest emergency department straight away.",
        "no_findings": "Your doctor did not select any findings to explain here.",
        "note_label": "Your doctor's note",
        "disclaimer": "This summary was made from a report your doctor reviewed and signed. "
        "It does not replace talking to your doctor. Ask your doctor if you have any questions.",
    },
    "hi": {
        "title": "आपकी छाती के एक्स-रे का सारांश",
        "none": "आपके डॉक्टर ने आपकी छाती का एक्स-रे देख लिया है। इस एक्स-रे के लिए आगे कुछ करने की ज़रूरत नहीं है।",
        "routine": "आपके डॉक्टर ने आपकी छाती का एक्स-रे देख लिया है और नियमित फ़ॉलो-अप की सलाह दी है।",
        "soon": "आपके डॉक्टर ने आपकी छाती का एक्स-रे देख लिया है और वे आपसे जल्द ही दोबारा मिलना चाहते हैं।",
        "urgent": "आपके डॉक्टर ने आपकी छाती का एक्स-रे देख लिया है और उन्हें आपसे तुरंत बात करनी है। "
        "कृपया आज ही अपने डॉक्टर या अस्पताल से संपर्क करें।",
        "step_weeks": "कृपया लगभग {weeks} सप्ताह में दोबारा जाँच करवाएँ।",
        "step_none": "यह सारांश अपने रिकॉर्ड के लिए रखें।",
        "step_routine": "अपने डॉक्टर की सलाह के अनुसार नियमित जाँच करवाएँ।",
        "step_soon": "कृपया जल्द ही अपने डॉक्टर से मिलने का समय लें।",
        "step_urgent": "आज ही अपने डॉक्टर से संपर्क करें। अगर साँस लेने में बहुत तकलीफ़ या छाती में दर्द हो, "
        "तो तुरंत नज़दीकी आपातकालीन विभाग जाएँ।",
        "no_findings": "आपके डॉक्टर ने यहाँ समझाने के लिए कोई निष्कर्ष नहीं चुना।",
        "note_label": "आपके डॉक्टर की टिप्पणी",
        "disclaimer": "यह सारांश उस रिपोर्ट से बनाया गया है जिसे आपके डॉक्टर ने जाँचकर हस्ताक्षर किया है। "
        "यह डॉक्टर से बात करने का विकल्प नहीं है। कोई भी प्रश्न हो तो अपने डॉक्टर से पूछें।",
    },
    "mr": {
        "title": "तुमच्या छातीच्या एक्स-रेचा सारांश",
        "none": "तुमच्या डॉक्टरांनी तुमचा छातीचा एक्स-रे तपासला आहे. या एक्स-रेसाठी पुढे काही करण्याची "
        "गरज नाही.",
        "routine": "तुमच्या डॉक्टरांनी तुमचा छातीचा एक्स-रे तपासला आहे आणि नियमित फॉलो-अपचा सल्ला दिला आहे.",
        "soon": "तुमच्या डॉक्टरांनी तुमचा छातीचा एक्स-रे तपासला आहे आणि त्यांना तुम्हाला लवकरच पुन्हा भेटायचे आहे.",
        "urgent": "तुमच्या डॉक्टरांनी तुमचा छातीचा एक्स-रे तपासला आहे आणि त्यांना तुमच्याशी तातडीने बोलायचे "
        "आहे. कृपया आजच तुमच्या डॉक्टरांशी किंवा रुग्णालयाशी संपर्क साधा.",
        "step_weeks": "कृपया सुमारे {weeks} आठवड्यांनी पुन्हा तपासणी करून घ्या.",
        "step_none": "हा सारांश तुमच्या नोंदीसाठी ठेवा.",
        "step_routine": "तुमच्या डॉक्टरांच्या सल्ल्यानुसार नियमित तपासणी करून घ्या.",
        "step_soon": "कृपया लवकरच तुमच्या डॉक्टरांची भेट घ्या.",
        "step_urgent": "आजच तुमच्या डॉक्टरांशी संपर्क साधा. श्वास घेण्यास खूप त्रास होत असेल किंवा छातीत "
        "दुखत असेल, तर लगेच जवळच्या आपत्कालीन विभागात जा.",
        "no_findings": "तुमच्या डॉक्टरांनी येथे समजावून सांगण्यासाठी कोणतेही निष्कर्ष निवडले नाहीत.",
        "note_label": "तुमच्या डॉक्टरांची टिप्पणी",
        "disclaimer": "हा सारांश तुमच्या डॉक्टरांनी तपासून स्वाक्षरी केलेल्या अहवालावरून तयार केला आहे. "
        "हा डॉक्टरांशी बोलण्याचा पर्याय नाही. काही प्रश्न असल्यास तुमच्या डॉक्टरांना विचारा.",
    },
}


def build_summary(
    language: str,
    patient_findings: list[str],
    follow_up: str,
    follow_up_weeks: int | None,
) -> dict[str, Any]:
    """Assemble the summary from the doctor's signed choices.

    Unknown labels are skipped rather than shown untranslated: the patient
    should never meet a raw model label name.
    """
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported language {language!r}; choose from {LANGUAGES}")
    t = TEXT[language]
    follow_up = follow_up if follow_up in ("none", "routine", "soon", "urgent") else "none"

    findings = [
        {
            "label": label,
            "name": FINDINGS[label][language][0],
            "explanation": FINDINGS[label][language][1],
        }
        for label in dict.fromkeys(patient_findings)
        if label in FINDINGS
    ]

    if follow_up == "urgent":
        next_steps = t["step_urgent"]
    elif follow_up_weeks:
        next_steps = t["step_weeks"].format(weeks=follow_up_weeks)
    else:
        next_steps = t[f"step_{follow_up}"]

    return {
        "title": t["title"],
        "headline": t[follow_up],
        "findings": findings,
        "no_findings_text": t["no_findings"],
        "next_steps": next_steps,
        "doctor_note_label": t["note_label"],
        "disclaimer": t["disclaimer"],
    }
