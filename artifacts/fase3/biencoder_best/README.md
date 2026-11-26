---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:4000
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
widget:
- source_sentence: '¿Qué banda de prioridad SOC y estado CISA KEV corresponden a esta
    vulnerabilidad en cisco/evolved_programmable_network_manager, asociada a CWE-20:
    «Cisco Prime Infrastructure 1.2.0 through 2.2(2) and Cisco Evolved Programmable
    Network Manager (EPNM) 1.2 allow remote attackers to execute arbitrary code via
    c»?'
  sentences:
  - CVE-2016-1291. KEV=no. banda=P3-plan. score=34.7. CVSS=9.8. CWE=CWE-20. tactica=Initial
    Access. producto=cisco/evolved_programmable_network_manager. Cisco Prime Infrastructure
    1.2.0 through 2.2(2) and Cisco Evolved Programmable Network Manager (EPNM) 1.2
    allow remote attackers to execute arbitrary code via crafted deserialized data
    in an HTTP POST request, aka Bug ID CSCuw03192.
  - CVE-2016-9272. KEV=no. banda=P3-plan. score=33.6. CVSS=9.1. CWE=CWE-89. tactica=Initial
    Access. producto=exponentcms/exponent_cms. A Blind SQL Injection Vulnerability
    in Exponent CMS through 2.4.0, with the rerank array parameter, can lead to site
    database information disclosure and denial of service.
  - CVE-2018-11635. KEV=no. banda=P3-plan. score=34.7. CVSS=9.8. CWE=CWE-798. tactica=Credential
    Access. producto=dialogic/powermedia_xms. Use of a Hard-coded Cryptographic Key
    used to protect cookie session data in /var/www/xms/application/config/config.php
    in the administrative console in Dialogic PowerMedia XMS through 3.5 allows remote
    attackers to bypass authentication.
- source_sentence: '¿Qué banda de prioridad SOC y estado CISA KEV corresponden a esta
    vulnerabilidad en techsolsystem/file_download_tracker, asociada a CWE-89: «SQL
    Injection exists in the File Download Tracker 3.0 component for Joomla! via the
    dynfield[phone] or sess parameter.»?'
  sentences:
  - CVE-2018-4991. KEV=no. banda=P3-plan. score=34.7. CVSS=9.8. CWE=CWE-295. tactica=Unknown.
    producto=adobe/creative_cloud. Adobe Creative Cloud Desktop Application versions
    4.4.1.298 and earlier have an exploitable Improper certificate validation vulnerability.
    Successful exploitation could lead to a security bypass.
  - CVE-2016-10176. KEV=no. banda=P3-plan. score=34.7. CVSS=9.8. CWE=CWE-20. tactica=Initial
    Access. producto=netgear/wnr2000v5_firmware. The NETGEAR WNR2000v5 router allows
    an administrator to perform sensitive actions by invoking the apply.cgi URL on
    the web server of the device. This special URL is handled by the embedded web
    server (uhttpd) and processed accordingly. The web server also contains another
    URL, apply_noauth.cgi, that allows an unauthenticated user to perform sensitive
    actions on the device. This functionality can be exploited to change the router
    settings (such as the answers to the password-recovery questions) and achieve
    remote code execution.
  - CVE-2018-6004. KEV=no. banda=P3-plan. score=34.7. CVSS=9.8. CWE=CWE-89. tactica=Initial
    Access. producto=techsolsystem/file_download_tracker. SQL Injection exists in
    the File Download Tracker 3.0 component for Joomla! via the dynfield[phone] or
    sess parameter.
- source_sentence: '¿Qué banda de prioridad SOC y estado CISA KEV corresponden a esta
    vulnerabilidad en qualcomm/sd_400_firmware, asociada a CWE-476: «In Android before
    2018-04-05 or earlier security patch level on Qualcomm Snapdragon Mobile SD 400,
    lack of address argument validation in qsee_get_tz_app_name()»?'
  sentences:
  - CVE-2016-1289. KEV=no. banda=P4-monitor. score=19.7. CVSS=9.8. CWE=CWE-119. tactica=Execution.
    producto=cisco/prime_infrastructure. The API in Cisco Prime Infrastructure 1.2
    through 3.0 and Evolved Programmable Network Manager (EPNM) 1.2 allows remote
    attackers to execute arbitrary code or obtain sensitive management information
    via a crafted HTTP request, as demonstrated by discovering managed-device credentials,
    aka Bug ID CSCuy10231.
  - CVE-2016-10489. KEV=no. banda=P3-plan. score=34.7. CVSS=9.8. CWE=CWE-476. tactica=Unknown.
    producto=qualcomm/sd_400_firmware. In Android before 2018-04-05 or earlier security
    patch level on Qualcomm Snapdragon Mobile SD 400, lack of address argument validation
    in qsee_get_tz_app_name() may lead to an untrusted pointer dereference.
  - CVE-2014-8888. KEV=no. banda=P4-monitor. score=19.7. CVSS=9.8. CWE=CWE-77. tactica=Execution.
    producto=dlink/dir-815_firmware. The remote administration interface in D-Link
    DIR-815 devices with firmware before 2.03.B02 allows remote attackers to execute
    arbitrary commands via vectors related to an "HTTP command injection issue."
- source_sentence: '¿Qué banda de prioridad SOC y estado CISA KEV corresponden a esta
    vulnerabilidad en python/python, asociada a CWE-190: «CPython (aka Python) up
    to 2.7.13 is vulnerable to an integer overflow in the PyString_DecodeEscape function
    in stringobject.c, resulting in heap-based buffer o»?'
  sentences:
  - CVE-2017-0359. KEV=no. banda=P3-plan. score=34.7. CVSS=9.8. CWE=CWE no indicado.
    tactica=Unknown. producto=reproducible_builds/diffoscope. diffoscope before 77
    writes to arbitrary locations on disk based on the contents of an untrusted archive.
  - CVE-2015-5254. KEV=no. banda=P4-monitor. score=19.7. CVSS=9.8. CWE=CWE-20. tactica=Initial
    Access. producto=redhat/openshift. Apache ActiveMQ 5.x before 5.13.0 does not
    restrict the classes that can be serialized in the broker, which allows remote
    attackers to execute arbitrary code via a crafted serialized Java Message Service
    (JMS) ObjectMessage object.
  - CVE-2017-1000158. KEV=no. banda=P3-plan. score=34.7. CVSS=9.8. CWE=CWE-190. tactica=Execution.
    producto=python/python. CPython (aka Python) up to 2.7.13 is vulnerable to an
    integer overflow in the PyString_DecodeEscape function in stringobject.c, resulting
    in heap-based buffer overflow (and possible arbitrary code execution)
- source_sentence: '¿Qué banda de prioridad SOC y estado CISA KEV corresponden a esta
    vulnerabilidad en tcpdump/tcpdump, asociada a CWE-125: «Several protocol parsers
    in tcpdump before 4.9.2 could cause a buffer over-read in addrtoname.c:lookup_bytestring().»?'
  sentences:
  - CVE-2017-12894. KEV=no. banda=P3-plan. score=34.7. CVSS=9.8. CWE=CWE-125. tactica=Collection.
    producto=tcpdump/tcpdump. Several protocol parsers in tcpdump before 4.9.2 could
    cause a buffer over-read in addrtoname.c:lookup_bytestring().
  - CVE-2016-6303. KEV=no. banda=P3-plan. score=34.7. CVSS=9.8. CWE=CWE-787. tactica=Execution.
    producto=nodejs/node.js. Integer overflow in the MDC2_Update function in crypto/mdc2/mdc2dgst.c
    in OpenSSL before 1.1.0 allows remote attackers to cause a denial of service (out-of-bounds
    write and application crash) or possibly have unspecified other impact via unknown
    vectors.
  - CVE-2018-13008. KEV=no. banda=P3-plan. score=34.7. CVSS=9.8. CWE=CWE-125. tactica=Collection.
    producto=gopro/gpmf-parser. An issue was discovered in gpmf-parser 1.1.2. There
    is a heap-based buffer over-read in GPMF_parser.c in the function GPMF_Next, related
    to certain checks for a positive nest_level.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for retrieval.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) <!-- at revision e8f8c211226b894fcb81acc59f3b34ba3efd5f42 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    '¿Qué banda de prioridad SOC y estado CISA KEV corresponden a esta vulnerabilidad en tcpdump/tcpdump, asociada a CWE-125: «Several protocol parsers in tcpdump before 4.9.2 could cause a buffer over-read in addrtoname.c:lookup_bytestring().»?',
    'CVE-2017-12894. KEV=no. banda=P3-plan. score=34.7. CVSS=9.8. CWE=CWE-125. tactica=Collection. producto=tcpdump/tcpdump. Several protocol parsers in tcpdump before 4.9.2 could cause a buffer over-read in addrtoname.c:lookup_bytestring().',
    'CVE-2018-13008. KEV=no. banda=P3-plan. score=34.7. CVSS=9.8. CWE=CWE-125. tactica=Collection. producto=gopro/gpmf-parser. An issue was discovered in gpmf-parser 1.1.2. There is a heap-based buffer over-read in GPMF_parser.c in the function GPMF_Next, related to certain checks for a positive nest_level.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.7332, 0.1684],
#         [0.7332, 1.0000, 0.1205],
#         [0.1684, 0.1205, 1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 4,000 training samples
* Columns: <code>anchor</code> and <code>positive</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                              | positive                                                                             |
  |:--------|:------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
  | type    | string                                                                              | string                                                                               |
  | details | <ul><li>min: 56 tokens</li><li>mean: 83.61 tokens</li><li>max: 128 tokens</li></ul> | <ul><li>min: 68 tokens</li><li>mean: 127.65 tokens</li><li>max: 256 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                                                                                                    | positive                                                                                                                                                                                                                                                                                                                                                                                                                                |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>¿Qué banda de prioridad SOC y estado CISA KEV corresponden a esta vulnerabilidad en tcpdump/tcpdump, asociada a CWE-125: «The IEEE 802.15.4 parser in tcpdump before 4.9.2 has a buffer over-read in print-802_15_4.c:ieee802_15_4_if_print().»?</code>                                                             | <code>CVE-2017-13000. KEV=no. banda=P3-plan. score=34.7. CVSS=9.8. CWE=CWE-125. tactica=Collection. producto=tcpdump/tcpdump. The IEEE 802.15.4 parser in tcpdump before 4.9.2 has a buffer over-read in print-802_15_4.c:ieee802_15_4_if_print().</code>                                                                                                                                                                               |
  | <code>¿Qué banda de prioridad SOC y estado CISA KEV corresponden a esta vulnerabilidad en cisco/nx-os, asociada a CWE-255: «Cisco NX-OS 7.0(1)N1(1), 7.0(1)N1(3), and 7.0(4)N1(1) on Nexus 2000 Fabric Extender devices has a blank root password, which allows local users to gain privileg»?</code>                     | <code>CVE-2016-1341. KEV=no. banda=P4-monitor. score=19.7. CVSS=9.8. CWE=CWE-255. tactica=Unknown. producto=cisco/nx-os. Cisco NX-OS 7.0(1)N1(1), 7.0(1)N1(3), and 7.0(4)N1(1) on Nexus 2000 Fabric Extender devices has a blank root password, which allows local users to gain privileges via unspecified vectors, aka Bug ID CSCur22079.</code>                                                                                      |
  | <code>¿Qué banda de prioridad SOC y estado CISA KEV corresponden a esta vulnerabilidad en dulwich_project/dulwich, asociada a CWE no indicado: «Dulwich before 0.18.5, when an SSH subprocess is used, allows remote attackers to execute arbitrary commands via an ssh URL with an initial dash character in th»?</code> | <code>CVE-2017-16228. KEV=no. banda=P3-plan. score=34.7. CVSS=9.8. CWE=CWE no indicado. tactica=Unknown. producto=dulwich_project/dulwich. Dulwich before 0.18.5, when an SSH subprocess is used, allows remote attackers to execute arbitrary commands via an ssh URL with an initial dash character in the hostname, a related issue to CVE-2017-9800, CVE-2017-12836, CVE-2017-12976, CVE-2017-1000116, and CVE-2017-1000117.</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Framework Versions
- Python: 3.13.9
- Sentence Transformers: 5.4.1
- Transformers: 5.6.2
- PyTorch: 2.6.0+cu124
- Accelerate: 1.16.0
- Datasets: 4.8.5
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{oord2019representationlearningcontrastivepredictive,
      title={Representation Learning with Contrastive Predictive Coding},
      author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
      year={2019},
      eprint={1807.03748},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1807.03748},
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->