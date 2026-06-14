# AU Reference

The Facial Action Coding System (FACS) is a system to taxonimize human facial movements based upon their underlying musculature. It was first developed by Carl-Herman Hjortsö in 1978 and then adopted and expanded upon by Paul Ekman and Wallace V. Friesen in 2002. The table below lists most of the possible Action Units (AUs) that are coded for along with with their muscular basis, expression involvement, and model support (not all models are trained to detect all AUs). Most models support a subset of of about 25 AUs corresponding specifically to *facial* muscles. Additional AUs are part of the FACS system but not commonly used to train models as they are associated with eye or head movement. 

You can sort the table by clicking on any column (shift+click to sort by multiple columns). You can also use the search bar (case insensitive) to filter the table by any of its content such as:
- which AUs are commonly associated with which expressions (e.g. search "happiness")
- which muscles are associated with which AUs (e.g. search "corrugator")
- which Py-Feat models support which AUs (e.g. search "svm")
- which **ARKit blendshape(s)** an AU roughly corresponds to (e.g. search "jawOpen")

The **ARKit Blendshape** column maps each AU to the closest of Apple's 52 ARKit
face blendshapes — useful when driving an avatar or comparing AU output to a
blendshape rig. The correspondence is **approximate**: FACS AUs and ARKit
blendshapes don't align one-to-one (some AUs have no blendshape equivalent, and
a few blendshapes span multiple AUs). Blendshape names follow the
[ARKit 52-blendshape reference](https://pooyadeperson.com/the-ultimate-guide-to-creating-arkits-52-facial-blendshapes/).

!!! note
    Emotion expressions (e.g. *sadness*) are often characterized by patterns of AUs (AU1 + AU4 + AU15), but **individuals vary** in the way the exact ways that they emote. For this reason, some AUs may not be observed on some faces for the same expression and some expressions may consists of *additional* AUs.

<link href="https://unpkg.com/gridjs/dist/theme/mermaid.min.css" rel="stylesheet" />
<div id="wrapper"></div>
<script src="https://unpkg.com/gridjs/dist/gridjs.umd.js"></script>

<script defer>
    const grid = new gridjs.Grid({
        search: true,
        sort: true,
        resizable: false,
        autoWidth: false,
        fixedHeader: true,
        height: '52rem',
        width: 'initial',
        style: {
            container: {
                'overflow': 'scroll'
            },
            table: {
                'font-size': '14px',
                'text-overflow': 'scroll',
            },
        },
        columns: [{name:'AU', formatter: (cell) => `AU${cell}`}, 'FACS Name', 'ARKit Blendshape', 'Muscles', 'FACS Category', 'Related Expression', 'Models', 'Notes'],
        data: [
            [1, 'Inner Brow Raiser', 'browInnerUp', 'Frontalis (medial)', 'main', 'sadness, surprise, fear', 'svm, xgb,  Py-Feat viz', ''],
            [2, 'Outer Brow Raiser', 'browOuterUpLeft, browOuterUpRight', 'Frontalis (lateral)', 'main', 'surprise, fear', 'svm, xgb,  Py-Feat viz', ''],
            [3, 'Inner corner Brow Tightener', '', 'Procerus, Depressor Supercilii, Corrugator Supercilii', 'extended (Baby FACS)', '', 'none', 'Only in babies! Analogue of AU4 in adults'],
            [4, 'Brow Lowerer', 'browDownLeft, browDownRight', 'Procerus, Depressor Supercilii, Corrugator Supercilii', 'main', 'sadness, fear, anger', 'svm, xgb,  Py-Feat viz', ''],
            [5, 'Upper Lid Raiser', 'eyeWideLeft, eyeWideRight', 'Levator Palpebrae Superioris, Superior Tarsal Muscle', 'main', 'surprise, fear, anger', 'svm, xgb, Py-Feat viz', ''],
            [6, 'Cheek Raiser', 'cheekSquintLeft, cheekSquintRight', 'Orbicularis Oculi (orbital)', 'main', 'happiness, disgust, contempt', 'svm, xgb, Py-Feat viz', ''],
            [7, 'Lid Tightener', 'eyeSquintLeft, eyeSquintRight', 'Orbicularis Oculi (palpebral)', 'main', 'fear, anger', 'svm, xgb,  Py-Feat viz', ''],
            [8, 'Lips Toward Each Other', 'mouthClose', 'Orbicularis Oris', 'main', 'none', '', ''],
            [9, 'Nose Wrinkler', 'noseSneerLeft, noseSneerRight', 'Levator Labii Superioris Alaeque Nasi', 'main', 'disgust', 'svm, xgb, Py-Feat viz', ''],
            [10, 'Upper Lip Raiser', 'mouthUpperUpLeft, mouthUpperUpRight', 'Levator Labii Superioris', 'main', '', 'svm, xgb,  Py-Feat viz', ''],
            [11, 'Nasolabial Deepener', '', 'Zygomaticus Minor', 'main', 'disgust, fear', 'svm, xgb, Py-Feat viz', ''],
            [12, 'Lip Corner Puller', 'mouthSmileLeft, mouthSmileRight', 'Zygomaticus Major', 'main', 'happiness, contempt', 'svm, xgb,  Py-Feat viz', ''],
            [13, 'Sharp Lip Puller', '', 'Levator Anguli Oris/Caninus', 'main', '', 'none', ''],
            [14, 'Dimpler', 'mouthDimpleLeft, mouthDimpleRight', 'Buccinator', 'main', 'contempt', 'svm, xgb,  Py-Feat viz', ''],
            [15, 'Lip Corner Depressor', 'mouthFrownLeft, mouthFrownRight', 'Depressor Anguli Oris', 'main', 'sadness, disgust', 'svm, xgb,  Py-Feat viz', ''],
            [16, 'Lower Lip Depressor', 'mouthLowerDownLeft, mouthLowerDownRight', 'Depressor Labii Inferioris', 'main', '', 'none', ''],
            [17, 'Chin Raiser', 'mouthShrugLower', 'Mentalis', 'main', 'disgust', 'svm, xgb,  Py-Feat viz', ''],
            [18, 'Lip Pucker', 'mouthPucker', 'Incisvii Labii Superioris, Incisvii Labii Inferioris', 'main', '', 'none', ''],
            [19, 'Tongue Show', 'tongueOut', 'Genioglossus, Medial Pterygoid, Masseter', 'main', '', 'none', ''],
            [20, 'Lip Stretcher', 'mouthStretchLeft, mouthStretchRight', 'Risorius, Platysma', 'main', 'fear', 'svm, xgb, Py-Feat viz', ''],
            [21, 'Neck Tightener', '', 'Platysma', 'main', '', 'none', ''],
            [22, 'Lip Funneler', 'mouthFunnel', 'Orbicularis Oris', 'main', '', 'none', ''],
            [23, 'Lip Tightener', 'mouthPressLeft, mouthPressRight', 'Orbicularis Oris', 'main', 'anger', 'svm, xgb,  Py-Feat viz', ''],
            [24, 'Lip Pressor', 'mouthPressLeft, mouthPressRight', 'Orbicularis Oris', 'main', '', 'svm, xgb,  Py-Feat viz', ''],
            [25, 'Lip Part', 'jawOpen (slight)', 'Depressor Labii Inferioris', 'main', 'happiness, surprise, fear', 'svm, xgb, Py-Feat viz', ''],
            [26, 'Jaw Drop', 'jawOpen', 'Masseter, Temporalis, Medial Pterygoid', 'main', 'fear, surprise', 'svm, xgb, Py-Feat viz', ''],
            [27, 'Mouth Stretch', 'jawOpen (max)', 'Pterygoids, Digastric', 'main', '', 'none', ''],
            [28, 'Lip Suck', 'mouthRollUpper, mouthRollLower', 'Orbicularis Oris', 'main', '', 'svm, xgb, Py-Feat viz', ''],
            [29, 'Jaw Thrust', 'jawForward', 'Pterygoids, Masseter', 'behavioral', '', 'none', ''],
            [30, 'Jaw Sideways', 'jawLeft, jawRight', 'Pterygoids, Masseter, Temporalis', 'behavioral', '', 'none', ''],
            [31, 'Jaw Clencher', '', 'Masseter', 'behavioral', '', 'none', ''],
            [32, 'Lip Bite', '', 'Masseter', 'behavioral', '', 'none', ''],
            [33, 'Cheek Blow', '', 'Buccinator, Orbicularis Oris, Mentalis', 'behavioral', '', 'none', ''],
            [34, 'Cheek Puff', 'cheekPuff', 'Buccinator, Orbicularis Oris, Mentalis, Depressor Depti Nasi', 'behavioral', '', 'none', ''],
            [35, 'Cheek Suck', '', 'Buccinator', 'behavioral', '', 'none', ''],
            [37, 'Lip Wipe', '', 'Pterygoids, Masseter, Genioglossus', 'behavioral', '', 'none', ''],
            [38, 'Nostril Dilator', '', 'Nasalis (alaris), Dilator Naris Anterior, Depressor Septi Nasi', 'behavioral', 'anger', 'none', ''],
            [39, 'Nostril Compressor', '', 'Nasalis (transverse), Compressor Narium Minor', 'behavioral', '', 'none', ''],
            [40, 'Sniff', '', '', 'behavioral', '', 'none', ''],
            [41, 'Lid Drop', '', 'Levator Palpebrae Superioris (relaxation)', 'behavioral', '', 'none', ''],
            [42, 'Slit', '', 'Depressor Supercilii', 'behavioral', '', 'none', 'Different muscular strand than AU4'],
            [43, 'Eyes Closed', 'eyeBlinkLeft, eyeBlinkRight', 'Levator Palebrae Superioris (relaxation)', 'behavioral', '', 'svm, xgb, Py-Feat viz', ''],
            [44, 'Squint', 'eyeSquintLeft, eyeSquintRight', 'Corrugator Supercilii', 'behavioral', '', 'none', 'Different muscular strand than AU4'],
            [45, 'Blink', 'eyeBlinkLeft, eyeBlinkRight', 'Levator Palebrae Superioris (relaxation), Orbicularis Oculi (contraction)', 'behavioral','', 'none', 'Different muscular strand than AU4'],
            [46, 'Wink', 'eyeBlinkLeft, eyeBlinkRight', 'Orbicularis Oculi', 'behavioral', '', 'none', ''],
            [51, 'Head turn left', '', '', 'head', '', 'none', ''],
            [52, 'Head turn right', '', '', 'head', '', 'none', ''],
            [53, 'Head up', '', '', 'head', '', 'none', ''],
            [54, 'Head down', '', '', 'head', '', 'none', ''],
            [55, 'Head tilt left', '', '', 'head', '', 'none', ''],
            [56, 'Head tilt right', '', '', 'head', '', 'none', ''],
            [57, 'Head forward', '', '', 'head', '', 'none', ''],
            [58, 'Head backward', '', '', 'head', '', 'none', ''],
            [61, 'Eyes turn left', 'eyeLookInLeft, eyeLookOutRight', 'Medial Rectus (right eye), Lateral Rectus (left eye)', 'eyes', '', 'none', ''],
            [62, 'Eyes turn right', 'eyeLookOutLeft, eyeLookInRight', 'Medial Rectus (left eye), Lateral Rectus (right eye)', 'eyes', '', 'none', ''],
            [63, 'Eyes up', 'eyeLookUpLeft, eyeLookUpRight', 'Superior Rectus, Inferior Oblique', 'eyes', '', 'none', ''],
            [64, 'Eyes down', 'eyeLookDownLeft, eyeLookDownRight', 'Inferior Rectus, Superior Oblique', 'eyes', '', 'none', ''],
            [65, 'Strabismus', '', '', 'eyes', '', 'none', 'Misaligned eyes when gazing'],
            [66, 'Cross-eyed', '', 'Medial Rectus', 'eyes', '', 'none', ''],
        ]
    });
    grid.render(document.getElementById('wrapper'))
</script>

<style>
    th {
        min-width: 175px !important;
    }
</style>