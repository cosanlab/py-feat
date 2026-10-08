<!-- Mirrored from LICENSES/COMMERCIAL-RETRAINING.md; keep the notices synchronized. -->

# Dataset options for commercially usable open weights

Reviewed October 8, 2026. Here, **commercially usable open weights** means
weights that may be publicly downloaded, redistributed, and used in
commercial applications. Publishing research-only weights is a different
target. This table assesses the evidence collected for Py-Feat; it does
not issue new permissions or determine whether a contract applies to a
particular party.

**No complete image/video training corpus in the current mixture has yet
been verified to permit both commercial-purpose training and public
commercial weight distribution.** FER+ has an MIT grant for its supplied
annotations and code, but that does not clear its separately sourced images.
Several other datasets have unanswered permission questions rather than an
established prohibition on every trained model. Those gaps are not a basis
for marking a dataset commercially cleared.

## Current and historical training sources

The commercial-training column concerns using the data to build a model
intended for commercial applications. The sharing column separately concerns
releasing the learned weights. A ban on redistributing dataset images is
not automatically a ban on distributing all trained parameters. Express
trained-model clauses and broader, unresolved clauses are distinguished.
See the [dataset register](licensing_datasets.md) for source links and acquisition
evidence; current website terms do not replace historical accepted terms.

| Dataset | Commercial-purpose training on reviewed terms | Public weight distribution | Retraining decision |
|---|---|---|---|
| BP4D | Outside the supplied noncommercial internal-research grant | Parameter-set/algorithm clause requires artifact-specific assessment | Obtain permission covering the intended training and release. |
| BP4D+ | Outside the supplied noncommercial grant; agreement term also needs resolution | Separate parameter-set/algorithm clause and renewal question | Obtain permission and resolve term before reuse. |
| DISFA | Research-only wording; no clear commercial-purpose grant established | No express weight-sharing grant or prohibition in the reviewed wording | Permission candidate; not commercially cleared. |
| EmotioNet | Approval expressly excludes for-profit institution use and product/service development | No public weight-sharing grant established | Separate commercial/product and release permission needed. |
| CK+ | Noncommercial research grant | No express trained-weight grant located | Separate permission needed. |
| UNBC-McMaster Shoulder Pain | Noncommercial and pain-purpose restrictions | No express trained-weight grant located | Permission must cover commercial use, project purpose, and release. |
| AM-FED | Historical lab EULA permits noncommercial research | Dataset redistribution barred; weight release not expressly addressed | Separate permission needed; existing model-training purpose is not commercial clearance. |
| Aff-Wild2 | Supplied academic agreements exclude commercial use | Newer EULA explicitly controls trained-model sharing through university-only access; operative historical version unresolved | Not suitable for unrestricted commercial release on reviewed grants. Resolve chronology and obtain separate rights. |
| AffectNet | Accepted 2021 terms limit use to noncommercial research/education | No express trained-weight distribution provision in accepted email | Separate commercial and release permission needed. |
| RAF-DB | Accepted request promises research use and no profit from use | No express weight grant in email; referenced historical webpage not fully recovered | Separate permission and original full terms needed. |
| FER+ annotations/code | MIT permits commercial use of these supplied materials | MIT covers those materials; does not establish image or resulting model rights by itself | Potential annotation component, conditional on a separately cleared image corpus. |
| FER2013 images | Original competition rules and source-image rights not fully verified | Unresolved | Recover original rules and image rights; do not rely on a mirror's license label. |
| ExpW | No explicit grant established from official download page | Unresolved | Obtain an explicit grant covering commercial training and open weight release. |
| MELD | Author-supplied code/annotation GPL terms are not noncommercial, but underlying TV audiovisual rights are unresolved | No model-specific grant found; GPL does not automatically attach to all trained weights | Not commercially cleared as a complete audiovisual corpus. Resolve source rights and release scope. |
| AFEW-VA | Provider restricts database to research purposes; no clear commercial-purpose grant | No express model grant; film rights also require consideration | Permission candidate; not commercially cleared. |
| ETH-XGaze | Access email expressly prohibits training models for commercial products | Additional dataset/software distribution limits; no express public learned-weight grant established | Separate permission needed. A generic CC badge omits additional conditions. |
| Gaze360 | Research license expressly excludes commercial applications of models trained on its data | Institutional-use and distribution limits require separate sharing assessment | Not suitable for commercial weights under reviewed research terms. |
| MPIIGaze | Noncommercial scientific use / CC BY-NC-SA 4.0 dataset terms | No express learned-weight grant; adapted-material scope unresolved | Separate permission needed; do not automatically assign ShareAlike to weights. |
| Columbia Gaze | Provider specifies noncommercial use | No express trained-weight grant located | Separate permission needed. |
| CelebV-HQ | Noncommercial research; commercial exploitation of derived data prohibited | No explicit trained-weight grant; derived-data scope unresolved | Resolve derived-data scope and obtain commercial/release permission. |
| JAFFE | Noncommercial scientific research | Dataset redistribution restricted; trained-weight release not expressly addressed | Separate permission needed; this trained the v1 emotion SVM. |
| DISFA+ | Accepted terms say research and development only; commercial-purpose scope unresolved | No express trained-weight provision | Permission candidate. It trained original visualization PLS, although it is not v2.8 gradient-training data. |

## Evaluation-only sources and other supplied agreements

Keeping restricted data out of gradient training does not, by itself,
authorize commercial benchmarking. Data used to choose checkpoints also
influences model development. For v2.8, DISFA+ and EYEDIAP were used in
checkpoint selection. This fact does not automatically relicense weights,
but a commercial training plan must address those uses explicitly.

| Dataset | Reviewed evidence relevant to a new commercial training/release plan |
|---|---|
| EYEDIAP | Supplied grant is noncommercial R&D in specified areas. Licensee ownership of results does not itself waive corpus restrictions or establish unrestricted distribution. |
| FacePlace | Current provider specifies CC BY-NC-SA 3.0 and noncommercial experimental/publication use. Not a cleared commercial training source. |
| 300W | Provider restricts research use and training commercial algorithms. Not a cleared source for commercial-purpose training. |
| WFLW | Official download page has no explicit license. Permission unresolved; availability alone is insufficient. |
| AFLW | Noncommercial research, commercial derived-data restriction, limited internal copying, and separate image-owner rights. Not commercially cleared. |
| MPIIFaceGaze | Excluded from v2.8 as duplicative of MPIIGaze. Needs its own acquisition/model-rights assessment for any future use. |
| BINED | Supplied scientific noncommercial grant; no general learned-weight grant located. |
| BioVid | Supplied academic/noncommercial pain-research conditions. Commercial training and release permission needed. |
| CASME2 | Research-only wording; no explicit commercial-purpose or trained-weight grant established. |
| CAS(ME)3 | Supplied agreement prohibits commercial use, including testing commercial systems. |
| MMI | Supplied agreement restricts academic use, commercial-system testing, and data redistribution. |
| Sayette GFT | Limited-term, study-specific noncommercial access. Resolve term and intended use; exemptions for specified metadata/baselines are not a blanket grant for videos or weights. |

Possession of these agreements does not establish that a dataset trained
any released model. Evaluation roles and unused agreements remain separate
from the training inventory.

## Backbone, teacher, and pipeline requirements

A commercially usable retrain needs cleared inputs beyond the primary
dataset. Reusing the current network and only changing its fine-tuning
images is insufficient.

| Component | Current constraint | Requirement for a commercial retrain |
|---|---|---|
| ConvNeXt-V2 initialization | Exact `timm/convnextv2_tiny.fcmae_ft_in22k_in1k` weights are CC BY-NC 4.0; code is MIT | Use separately cleared pretrained weights or initialize independently. Do not initialize from the current Py-Feat checkpoint. |
| img2pose teacher / pose MLP | img2pose is CC BY-NC 4.0; student-weight scope needs assessment | Clear teacher use and target/student distribution or replace this supervision source. |
| MediaPipe geometry targets | Separate third-party source/model terms | Verify the exact teacher assets and intended use; software licensing alone is insufficient provenance. |
| RetinaFace and identity/landmark assets | Separate model/data grants; ArcFace weights are noncommercial research; several landmark grants remain unresolved | Audit or replace each shipped asset. Disabling ArcFace alone does not clear the whole pipeline. |
| Visualization/regression mappings | Existing fitted artifacts use prior detector/teacher predictions and CelebV-HQ | Refit from cleared sources and targets, or establish rights for existing artifacts. |

The practical first step is to assemble a corpus with documented permission
for **commercial model development and public redistribution of trained
weights**, together with appropriate source-image rights. A separately
licensed or newly collected corpus may be needed. Preserve the exact terms,
dates, sources, and checkpoint manifests so the new release can carry a
defensible permissive license. The current review has not identified a
verified subset of the existing complete corpora that already meets that
standard without further rights work.
