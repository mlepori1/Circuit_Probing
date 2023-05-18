# Summary

This repository includes the Dependency Treebank of Spoken L2 English (SL2E), which consists of Universal Dependency annotations for a random sample of sentences from the <a href="https://alaginrc.nict.go.jp/nict_jle/index_E.html" target="_blank">NICT JLE</a>, a corpus of spoken second language English. <a href="https://github.com/LCR-ADS-Lab/SL2E-Dependency-Treebank" target="_blank">The homepage of the project is here.</a>


# Introduction
This treebank is a part of a larger effort to make more treebank data that represents second language (L2) use publicly available. 

Fine-grained part of speech tags (XPOS) were based on the Penn Treebank Tagset and were manually annotated from scratch by at least two annotators. The XPOS annotation guidelines <a href="https://kristopherkyle.github.io/L2-Annotation-Project/anno_overview.html" target="_blank">can be found here</a>. Dependency annotations followed the Universal Dependencies (version 2.0) and were also manually annotated from scratch by at least two trained annotators. Supplemental annotation guidelines <a href="https://kristopherkyle.github.io/L2-Annotation-Project/dep_anno_overview.html" target="_blank">can be found here</a>. 

Universal part of speech tags (UPOS) were added using a probabilistic model trained using both L1 and L2 sections of the English UD treebanks (EWT, GUM, GUMReddit, Pronouns, PUD, and ESL) that relies on the XPOS tag and the UD syntactic relation. Automatic tagging accuracy was .9885 (macro accuracy), and all tags acheived an f1 above .97. The UPOS tags were subsequently checked manually.

More data is available that has been tagged with XPOS tags only on <a href="https://github.com/LCR-ADS-Lab/SL2E-Dependency-Treebank" target="_blank">the project homepage</a>.

# Acknowledgments
We would like to thank all of the undergraduate linguists who have contributed to this project: Chasen Afghani, Charles Baker-Rozell, Tyler Demmon, Zoe Haupt, Reed Jordan, Aaron Miller, Ted Sither, Grace Teuscher, and Claire Worthington

## References

**Treebank reference:**

Kyle, K., Eguchi, M., Miller, A., & Sither, T. (2022). *A dependency treebank of spoken second language English*. In *Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Innovative Use of NLP for Building Educational Applications (BEA 2022)*, pp. 39-45., Seattle, USA. Association for Computational Linguistics. <a href="https://aclanthology.org/2022.bea-1.7.pdf" target="_blank">pdf</a>

**Source Corpus Reference:**

Izumi, E., Uchimoto, K., & Isahara, H. (2004). The NICT JLE Corpus: Exploiting the language learners’ speech database for research and education. *International Journal of The Computer,the Internet and Management, 12*(2), 119–125.

# Changelog

* 2023-05-15 v2.12
  * Initial release in Universal Dependencies.


<pre>
=== Machine-readable metadata (DO NOT REMOVE!) ================================
Data available since: UD v2.12
License: CC BY-SA 4.0
Includes text: yes
Genre: spoken
Lemmas: not available
UPOS: converted with corrections
XPOS: manual native
Features: manual native
Relations: manual native
Contributors: Kyle, Kris; Eguchi, Masaki; Miller, Aaron; Sither, Ted
Contributing: here
Contact: kkyle2@uoregon.edu
===============================================================================
</pre>
