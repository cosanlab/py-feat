<!-- Mirrored from LICENSES/DATASETS.md; keep the notices synchronized. -->

# Dataset provenance and permission register

Reviewed October 8, 2026. This register separates **dataset terms** from
**licenses for software and pretrained weights**. It is not a sublicense of
any dataset and does not assume that every dataset condition automatically
becomes a condition on all learned parameters or inference outputs.

Public website terms describe the versions available at review time; they
do not establish which historical version was accepted. Dated access
records are distinguished below.

The v1 and v2 names identify model families, not package version numbers.
V2 roles below refer to `face_multitask_v28.safetensors`; older checkpoints
need separate provenance. Training includes pretraining, distillation, and
fine-tuning. Held-out data used for checkpoint selection are identified
separately from gradient training.

## Detector training data

| Dataset | Relevant artifact / role | Dataset terms and evidence | Remaining scope question |
|---|---|---|---|
| BP4D | v1 AU; original visualization PLS; v2 stages 1-3 | [Provider](https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html); signed agreement: noncommercial internal research, evaluation, teaching; restricted redistribution | Agreement addresses parameter sets and algorithms that substantially describe or reflect database information. Applicability to each artifact's distribution needs resolution. |
| BP4D+ | v1 AU supplement; v2 stages 1-3 | Same [provider](https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html); separate signed agreement and term | Parameter-set clause and renewal status must be checked separately from BP4D. |
| DISFA | v1 AU; v2 stages 1-3 | [Provider form](https://mohammadmahoor.com/pages/databases/disfa/forms/disfa/); signed agreement and accepted 2021 terms: research only, no onward video distribution | Signed wording does not itself state a separate noncommercial weight license. |
| EmotioNet | Original visualization PLS; v2 stages 1-3; listed in v1 supplement's training section | [Original access form](https://cbcsl.ece.ohio-state.edu/dbform_emotionet.html); 2021 access approval: institutional research only, no for-profit institution use, no product/service development or dataset redistribution | Source-image rights and trained-weight distribution scope require separate review; v1 AU classifier involvement needs an exact manifest. |
| CK+ | v1 AU/emotion; v2 stages 1-3 | [Provider agreement](https://ckplus.jeffcohn.net/request); signed agreement: noncommercial research, no dataset redistribution, subject-specific publication rules | No express trained-weight distribution grant located. |
| UNBC-McMaster Shoulder Pain / PAIN | v1 AU; v2 stages 1-3 | [Provider agreement](https://painarchive.jeffcohn.net/request); supplied PAINFUL agreement: noncommercial research/teaching and specified pain-research purposes | Confirm project-purpose coverage for generic multitask training and model sharing. |
| AM-FED | v2 stages 2-3 AU | [Authors' paper, section 8](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W16/papers/McDuff_Affectiva-MIT_Facial_Expression_2013_CVPR_paper.pdf); recovered 2016 lab EULA: noncommercial research, no dataset redistribution, publication/citation conditions | The stated project includes training an automatic FACS model. No express public trained-weight grant; confirm scope for each later artifact. |
| Aff-Wild2 | Current v1 AU documentation; v2 stages 2-3 AU and V/A | [Provider](https://sites.google.com/view/dimitrioskollias/databases/aff-wild2); two supplied 2026 agreements differ materially; the original acquisition agreement has not been located | Newer agreement explicitly requires controlled university-only access for covered trained models and downstream derivatives. Check which agreement governs each release; do not presume retroactivity or unrestricted rights under the earlier agreement. |
| AffectNet | v1 emotion evaluation; v2 stages 2-3 emotion/V-A training and held-out evaluation | [Provider](https://mohammadmahoor.com/pages/databases/affectnet/); accepted 2021 terms: noncommercial research/education, institutional storage, no onward dataset sharing | The accepted terms do not expressly settle trained-weight redistribution. Current AffectNet+ terms are not a substitute for the historical agreement. |
| RAF-DB | v2 stages 2-3 emotion; held-out test split | [Provider endpoint](http://www.whdeng.cn/raf/model1.html); 2021 accepted request/approval: research only, no third-party availability, sale or profit from use | Accepted email references additional webpage terms whose historical full text has not been recovered. No express public trained-weight grant in the email. |
| FER+ / FER2013 | FER+ v2 stages 2-3 emotion; FER2013 upstream ResMaskNet training | [Microsoft FER+](https://github.com/microsoft/FERPlus): MIT for supplied labels/software; images obtained separately | The [original Kaggle data page](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) identifies competition-rule terms; the full original rules and underlying image rights remain unresolved. |
| ExpW | v1 emotion SVM; v2 stages 2-3 emotion | [Provider project](https://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html) | No explicit general license established from the retrieved project page. |
| MELD | v2 stages 2-3 emotion; held-out validation/benchmark | [Author dataset](https://huggingface.co/datasets/declare-lab/MELD): GPL-3.0 label; audiovisual source is *Friends* | Separate supplied annotation/code licensing from underlying audiovisual rights; no automatic GPL licensing conclusion for weights. |
| AFEW-VA | v2 stages 2-3 V/A; held-out validation | [Provider](https://ibug.doc.ic.ac.uk/resources/afew-va-database/): research purposes | Film/source rights and model redistribution scope require separate review. |
| ETH-XGaze | v2 stages 2-3 gaze; held-out test split | [Provider](https://ait.ethz.ch/xgaze): CC BY-NC-SA 4.0 with additional conditions; 2026 access email expressly prohibits training models for commercial products | The [current additional-terms template](https://drive.google.com/file/d/1zwlDcP3W-feooegiXpGKeeeAyIPctLW_/view) restricts dataset/software distribution. Historical accepted version and trained-weight distribution scope need confirmation; no automatic weight ShareAlike conclusion. |
| Gaze360 | v2 stages 2-3 gaze; held-out splits; upstream shipped L2CS checkpoint | [Provider license](https://github.com/erkil1452/gaze360/blob/master/LICENSE.md): research only; explicitly excludes commercial applications of models trained on the dataset; limits use and redistribution of licensed material | This is express trained-model language, not assumed inheritance. Establish applicable acquisition terms and public weight-sharing permission separately from L2CS MIT code. |
| MPIIGaze | v2 stages 2-3 gaze; held-out subjects | [Provider](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild): CC BY-NC-SA 4.0, noncommercial scientific purposes | Dataset terms do not alone determine the checkpoint's license. |
| Columbia Gaze | v2 stages 2-3 gaze; held-out test split | [Provider](https://cave.cs.columbia.edu/repository/ColumbiaGazeDataSet): noncommercial use and citation | No standalone trained-weight distribution grant established. |
| CelebV-HQ | v2 stage 1; pose MLP; later visualization/regression mappings | [Provider agreement](https://celebv-hq.github.io/#agreement): noncommercial research; restricted dataset distribution and commercial exploitation of derived data | Clarify trained-weight and teacher-output scope for each derived artifact. |
| JAFFE | v1 emotion SVM training | [Authors' dataset record](https://zenodo.org/records/14974867): noncommercial scientific research and restricted redistribution | Not a validation-only dataset for the emotion SVM. |

## Evaluation, upstream pretraining, and additional supplied agreements

| Dataset | Role / evidence boundary | Terms or review status |
|---|---|---|
| DISFA+ | Held-out v1 AU evaluation; **training for original visualization PLS**; no v2.8 gradient training, but used in checkpoint selection | [Provider form](https://mohammadmahoor.com/pages/databases/disfa_plus/forms/disfa_plus/); separately accepted 2021 DISFA+ terms: research/development, citation, no dataset redistribution. No express trained-weight grant in that wording. |
| EYEDIAP | No v2.8 gradient training; benchmark/checkpoint selection | Supplied agreement: noncommercial R&D in specified areas, restricted corpus distribution, and express licensee ownership of research results. Result ownership and corpus-derivative scope are separate issues. |
| FacePlace | V2.8 evaluation; no gradient training in the current manuscript | [Provider](https://sites.google.com/andrew.cmu.edu/tarrlab/stimuli): CC BY-NC-SA 3.0, noncommercial experimental/publication use and acknowledgments. Historical accepted version not established. |
| WFLW | V2.8 evaluation; no gradient training in the current manuscript | [Official download page](https://wywu.github.io/projects/LAB/WFLW.html) provides files and citation but no explicit license; terms remain unresolved. |
| AFLW | V2.8 evaluation; no gradient training in the current manuscript | [Provider agreement](https://www.tugraz.at/institute/icg/research/team-bischof/learning-recognition-surveillance/downloads/aflw): noncommercial research, no commercial exploitation of derived data, limited internal copying; Flickr image-owner rights are separately reserved. |
| 300W | V1 landmark benchmark; v2.8 evaluation; upstream landmark-training mixtures require separate manifests | [Provider](https://ibug.doc.ic.ac.uk/resources/300-W/): research use and restrictions on training commercial algorithms. |
| MPIIFaceGaze | Excluded from v2.8 as a duplicate of MPIIGaze; separate upstream L2CS checkpoint option | Do not confuse it with the shipped Gaze360 L2CS checkpoint or assume joint training. |
| WIDER FACE | RetinaFace/img2pose upstream training and separate benchmark splits | [Provider](https://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/): CC BY-NC-ND dataset terms; not permissive data solely because detector code is MIT. |
| ImageNet-1K / ImageNet-22K | Upstream v2 backbone pretraining | The [exact pretrained backbone](https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in22k_in1k) is CC BY-NC 4.0; distinguish the checkpoint grant from image rights and MIT implementation code. |
| VGGFace2 / WebFace600K | FaceNet / ArcFace upstream identity training | Preserve exact checkpoint provenance and provider model terms; ArcFace pretrained models are noncommercial research. |
| BINED | Supplied agreement; not identified as training in the reviewed v1 detector or v2.8 sources | Scientific noncommercial use; restricted redistribution. No general trained-weight clause located. |
| BioVid | Supplied agreement; no training role established here | Academic/noncommercial pain research; restricted data access and distribution. |
| CASME2 | Supplied agreement; no training role established here | Research-only images/videos, no onward dataset provision. Does not expressly give a noncommercial weight license. |
| CAS(ME)3 | Supplied agreement; no training role established here | Commercial use, including commercial-system testing, prohibited; restricted redistribution and image publication. |
| MMI | Supplied agreement; no training role established here | Academic research; commercial-system testing and data redistribution restricted. |
| Sayette GFT | Supplied agreement; no training role established here | Limited-term noncommercial study-specific access; renewal status needs review. The agreement separately exempts specified metadata/baseline results from licensing. |

Possessing an agreement is not evidence that the corresponding dataset
trained a released model. Absence from a training inventory is likewise not
proof it was never used for evaluation, selection, or another artifact.
The original paper also evaluates on BIWI and NAMBA; those benchmark uses
must not be silently turned into training claims.

## How to read permission gaps

The [commercial retraining table](licensing_commercial.md) separates
commercial-purpose training from public weight sharing for each source.

“Not established” means the reviewed evidence does not support a definitive
grant; it is not a declaration that all use is prohibited. Dataset-access
agreements can impose obligations on the researcher without automatically
relicensing every output. Conversely, ownership of trained parameters or an
MIT implementation does not waive an applicable agreement's express model
sharing restrictions. No new grant or retroactive revocation is made here.

Signed agreements, signatures, private correspondence, and the unpublished
manuscript are maintained outside this public repository. Institutional
review should resolve the affected checkpoint permissions, agreement
chronology/renewals, and missing upstream grants before a broader model
license or unrestricted-distribution claim is made.
