# Summary

Repository for the Genre Tests for Linguistic Evaluation (GENTLE) Corpus

# Introduction

This repository contains release versions of the Genre Tests for Linguistic Evaluation (GENTLE) corpus, an English out-of-domain test set following the same multilayer annotations found in the [GUM corpus](https://gucorpling.org/gum/). The texts are of the following 8 genres:

  * dictionary entries
  * live esports commentary
  * legal documents
  * medical notes
  * poetry
  * mathematical proofs
  * course syllabuses
  * threat letters

# Additional annotations in MISC

The MISC column contains **entity, coreference, information status, Wikification and discourse** annotations from the full GENTLE corpus, encoded using the annotations `Entity`, `SplitAnte`, `Bridge` and `Discourse`. 

## Entity

The `Entity` annotation uses the CoNLL 2012 shared task bracketing format, which identifies potentially coreferring entities using round opening and closing brackets as well as a unique ID per entity, repeated across mentions. In the following example (taken from UD_English-GUM, which follows the same format), actor Jared Padalecki appears in a single token mention, labeled `(1-person-giv:act-cf2*-1-coref-Jared_Padalecki)` indicating the entity type (`person`) combined with the unique ID of all mentions of Padalecki in the text (`1-person`). Because Padalecki is a named entity with a corresponding Wikipedia page, the Wikification identifier corresponding to his Wikipedia page is given after the last hyphen (`1-person-Jared_Padalecki`). We can also see an information status annotation (`giv:act`, indicating an aforementioned or 'given' entity, actively mentioned last no farther than the previous sentences; see Dipper et al. 2007), a Centering Theory annotation (`cf2*`, indicating he is the second most central salient entity in the sentence moving forward, and that he was mentioned in the previous sentence, indicated by the `*`), as well as minimum token ID information indicating the head tokens for fuzzy matching (in this case `1`, the first and only token  in this span) and the coreference type `coref`, indicating lexical subsequent mention. The labels for each part of the hyphen-separated annotation are given at the top of each document in a comment `# global.Entity = GRP-etype-infstat-centering-minspan-link-identity`, indicating that these annotations consist of the entity group id (i.e the coreference group), entity type, information status, centering theory annotation, minimal span of tokens for head matching, the coreference link type, and named entity identity (if available). 

Multi-token mentions receive opening brackets on the line in which they open, such as `(97-person-giv:inact-cf4-1,3-coref-Jensen_Ackles`, and a closing annotation `97)` at the token on which they end. Multiple annotations are possible for one token, corresponding to nested entities, e.g. `(175-time-giv:inact-cf5-1-coref)189)` below corresponds to the single token and last token of the time entities "2015" and "April 2015" respectively. 

```CoNLL-U
# global.Entity = GRP-etype-infstat-centering-minspan-link-identity
...
1	For	for	ADP	IN	_	4	case	4:case	Discourse=joint-sequence_m:104->98:2
2	the	the	DET	DT	Definite=Def|PronType=Art	4	det	4:det	Bridge=173<188|Entity=(188-event-acc:inf-cf6-3,6,8-sgl
3	second	second	ADJ	JJ	Degree=Pos|NumType=Ord	4	amod	4:amod	_
4	campaign	campaign	NOUN	NN	Number=Sing	16	obl	16:obl:for	_
5	in	in	ADP	IN	_	10	case	10:case	_
6	the	the	DET	DT	Definite=Def|PronType=Art	10	det	10:det	Entity=(173-abstract-giv:inact-cf3-2,4,5-coref
7	Always	Always	ADV	NNP	Number=Sing	8	advmod	8:advmod	XML=<hi rend:::"italic">
8	Keep	Keep	PROPN	NNP	Number=Sing	10	compound	10:compound	_
9	Fighting	Fighting	PROPN	NNP	Number=Sing	8	xcomp	8:xcomp	XML=</hi>
10	series	series	NOUN	NN	Number=Sing	4	nmod	4:nmod:in	Entity=173)
11	in	in	ADP	IN	_	12	case	12:case	_
12	April	April	PROPN	NNP	Number=Sing	4	nmod	4:nmod:in	Entity=(189-time-new-cf10-1-sgl|XML=<date when:::"2015-04">
13	2015	2015	NUM	CD	NumForm=Digit|NumType=Card	12	nmod:tmod	12:nmod:tmod	Entity=(175-time-giv:inact-cf5-1-coref)189)188)|SpaceAfter=No|XML=</date>
14	,	,	PUNCT	,	_	4	punct	4:punct	_
15	Padalecki	Padalecki	PROPN	NNP	Number=Sing	16	nsubj	16:nsubj	Entity=(1-person-giv:act-cf2*-1-coref-Jared_Padalecki)
16	partnered	partner	VERB	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	0	root	0:root	_
17	with	with	ADP	IN	_	18	case	18:case	_
18	co-star	co-star	NOUN	NN	Number=Sing	16	obl	16:obl:with	Entity=(97-person-giv:inact-cf4-1,3-coref-Jensen_Ackles
19	Jensen	Jensen	PROPN	NNP	Number=Sing	18	appos	18:appos	XML=<ref target:::"https://en.wikipedia.org/wiki/Jensen_Ackles">
20	Ackles	Ackles	PROPN	NNP	Number=Sing	19	flat	19:flat	Entity=97)|XML=</ref>
21	to	to	PART	TO	_	22	mark	22:mark	Discourse=purpose-goal:105->104:0
22	release	release	VERB	VB	VerbForm=Inf	16	advcl	16:advcl:to	_
23	a	a	DET	DT	Definite=Ind|PronType=Art	24	det	24:det	Entity=(190-object-new-cf7-2-coref
24	shirt	shirt	NOUN	NN	Number=Sing	22	obj	22:obj	Entity=190)
25	featuring	feature	VERB	VBG	VerbForm=Ger	24	acl	24:acl	Discourse=elaboration-attribute:106->105:0
26	both	both	DET	DT	PronType=Art	25	obj	25:obj	Entity=(191-object-new-cf9-1-sgl
27	of	of	ADP	IN	_	29	case	29:case	_
28	their	their	PRON	PRP$	Case=Gen|Number=Plur|Person=3|Poss=Yes|PronType=Prs	29	nmod:poss	29:nmod:poss	Entity=(192-person-acc:aggr-cf1-1-coref)|SplitAnte=1<192,97<192
29	faces	face	NOUN	NNS	Number=Plur	26	nmod	26:nmod:of	Entity=191)|SpaceAfter=No
```

Possible values for the annotations mentioned above are:

  * entity type: abstract, animal, event, object, organization, person, place, plant, substance, time
  * information status 
    * new - not previously mentioned
    * giv:act - mentioned no further than one sentence ago
    * giv:inact - mentioned earlier
    * acc:inf - accessible, inferable from some previous mention (e.g. the house... [the door]) 
    * acc:aggr - accessible, aggregate, i.e. split antecedent mediated by a set of previous mentions
    * acc:com - accessible, common ground, i.e. generic ([the world]) or situationally accessible ("pass [the salt]", first mention of "you" or "I")
  * centering: 
    * rank in the forward looking center (Cf), and a '*' for the top entity also mentioned in the previous sentence (Cb). The preferred forward looking center (Cp) is simply expressed as cf1. 
    * centering transition types are computed from these annotations in the sentence level `# transition` annotations
  * link: 
    * ana - pronominal anaphora (the dancers ... [they])
    * appos - apposition (Kim, [the lawyer])
    * cata - cataphora ("In [their] speech, the athletes said", or expletive cataphora: "[it] is easy [to dance]")
    * coref - lexical coreference (e.g. [Kim] ... [Kim])
    * disc - discourse deixis, non-NP, e.g. verbal antecedent as in "[Kim arrived] - [this] delighted the children
    * pred - predication, e.g. Kim is [a teacher] (but NOT definite identification: This is Kim)
    * sgl - singleton, not mentioned again in document
  * identity: any Wikipedia article title
  * minspan: a number or set of comma-separated numbers indicating indices of minimal head tokens within the span of the mention (first in span: 1, etc.)

## Split antecedent and bridging

The annotations `SplitAnte` and `Bridge` mark non-strict identity anaphora (see the [Universal Anaphora](http://universalanaphora.org/) project for more details). For example, at token 28 in the example, the pronoun "their" refers back to two non-adjacent entities, requiring a split antecedent annotation. The value `SplitAnte=1<192,97<192` indicates that `192-person` (the pronoun "their") refers back to two previous Entity annotations, with pointers separatated by a comma: `1` (`1-person-...Jared_Padalecki`) and `97` (`97-person-...Jensen_Ackles`). 

Bridging anaphora is annotated when an entity has not been mentioned before, but is resolvable in context by way of a different entity: for example, token 2 has the annotation `Bridge=173<188`, which indicates that although `188-event` ("the second campaign...") has not been mentioned before, its identity is mediated by the previous mention of another entity, `173-abstract` (the project "Always Keep Fighting", mentioned earlier in the document, to which the campaign event belongs). In other words, readers can infer that "the second campaign" is part of the already introduced larger project, which also had a first campaign. This inference also leads to the information status label `acc:inf`, accessible-inferable.

## RST discourse trees

Discourse annotations are given in RST dependencies following the conversion from RST constituent trees as suggested by Li et al. (2014) - for the original RST constituent parses of GENTLE see the [source repo](https://github.com/gucorpling/gentle/). At the beginning of each Elementary Discourse Unit (EDU), an annotation `Discourse` gives the discourse function of the unit beginning with that token, followed by a colon, the ID of the current unit, and an arrow pointing to the ID of the parent unit in the discourse parse. For instance, `Discourse=purpose-goal:105->104:0` at token 21 in the example below means that this token begins discourse unit 105, which functions as a `purpose-goal` to unit 104, which begins at token 1 in this sentence ("Padalecki partnered with co-star Jensen Ackles --purpose-goal-> to release a shirt..."). The final `:0` indicates that the attachment has a depth of 0, without an intervening span in the original RST constituent tree (this information allows deterministic reconstruction of the RST constituent discourse tree from the conllu file). The unique `ROOT` node of the discourse tree has no arrow notation, e.g. `Discourse=ROOT:2:0` means that this token begins unit 2, which is the Central Discourse Unit (or discourse root) of the current document. Although it is easiest to recover RST constituent trees from the source repo, it is also possible to generate them automatically from the dependencies with depth information, using the scripts in the [rst2dep repo](https://github.com/amir-zeldes/rst2dep/).

Discourse relations in GENTLE are defined based on the effect that W (a writer/speaker) has on R (a reader/hearer) by modifying a Nucleus discourse unit (N) with another discourse unit (a Satellite, S, or another N). Discourse relation units can precede their nuclei (satellite-nucleus, or SN relation), follow them (NS), or be coordinated with each other (NN or multinuclear relations). Relations are classified hierarchically into 15 major classes and include:

  * Adversative
    * adversative-antithesis (SN/NS) - R is meant to prefer N as an alternative to S
    * adversative-concession (SN/NS) - R is meant to look past an incompatibility of N with S
    * adversative-contrast (NN) - W presents multiple Ns as incompatible, but of equal prominence
  * Attribution
    * attribution-positive (SN/NS) - S states a source for the information in N
    * attribution-negative (SN/NS) - S states that a potential source is NOT a source of the information in N
  * Causal
    * causal-cause (SN/NS) - S is the cause of N (and N is more prominent)
    * causal-result (SN/NS) - S is the result of N (or: N is the cause of S, and N is more prominent)
  * Context
    * context-background (SN/NS) - S provides prerequisite information to increase R's understanding of N
    * context-circumstance (SN/NS) - S details circumstances (often spatio-temporal) under which N applies
  * Contingency
    * contingency-condition (SN/NS) - N occurs (or not) depending on S
  * Elaboration
    * elaboration-attribute (NS) - S gives additional information about a participant within N (not on the entire proposition in N)
    * elaboration-additional (NS) - S gives additional information about the proposition in N as a whole
  * Explanation
    * explanation-evidence (SN/NS) - S provides evidence which increases R's belief in N
    * explanation-justify (SN/NS) - S increases R's acceptance of W's right to say N
    * explanation-motivation (SN/NS) - S is meant to influence R's willingness to act according to N
  * Evaluation
    * evaluation-comment (SN/NS) - S provides an assessment of N by W (R does not have to share this assessment)
  * Joint
    * joint-disjunction (NN) - W presents multiple Ns which can be regarded as interchangeable alternatives
    * joint-list (NN) - W presents multiple Ns in parallel which are additive, of equal prominence, and of equivalent purpose
    * joint-sequence (NN) - Multiple Ns form a temporally ordered sequence of events presented in chronological order
    * joint-other (NN) - a collection of unlike Ns of equal prominence, but of disparate (non-equivalent) discourse purpose
  * Mode
    * mode-manner (SN/NS) - S indicates the manner in which N happens
    * mode-means (SN/NS) - S indicates the means by which N happens
  * Organization
    * organization-heading (SN) - S prepared R for N using an explicit text organizing device such as a heading
    * organization-phatic (SN/NS) - S prepares R for N by holding the floor for W, without contributing propositional content
    * organization-preparation (SN) - covers all other forms of S units primarily used to signal an upcoming N
  * Purpose
    * purpose-attribute (SN/NS) - S gives the purpose of a participant within N (not the entire propostion in N)
    * purpose-goal (SN/NS) - the proposition in N as a whole is initiated or exists in order to realize S
  * Restatement
    * restatement-partial (NS) - S partly realizes the same role and content as a previous N
    * restatement-repetition (NN) - Multiple Ns of equal prominence realize the same role and content
  * Topic
    * topic-question (SN) - S steers the discourse topic by posing a question to which N is the answer
    * topic-solutionhood (SN/NS) - S steers the discourse topic by posing a problem, to which N presents a solution
  * Same-unit (NN) - connects parts of a discontinuous discourse unit (this is not a discourse relation)

## XML

Markup from the original XML annotations using TEI tags is available in the XML MISC annotation, which indicates which XML tags, if any, were opened or closed before or after the current token, and in what order. In tokens 7-9 in the example above, the XML annotations indicate the words "Always Keep Fighting" were originally italicized using the tag pair `<hi rend="italic">...</hi>`, which opens at token 7 and closes after token 9. To avoid confusion with the `=` sign in MISC annotations, XML `=` signs are escaped and represented as `:::`.

```CoNLL-U
7	Always	Always	ADV	NNP	Number=Sing	8	advmod	8:advmod	XML=<hi rend:::"italic">
8	Keep	Keep	PROPN	NNP	Number=Sing	10	compound	10:compound	_
9	Fighting	Fighting	PROPN	NNP	Number=Sing	8	xcomp	8:xcomp	XML=</hi>
```

XML block tags spanning whole sentences (i.e. not beginning or ending mid sentence), such as paragraphs (`<p>`) or headings (`<head>`) are instead represented using the standard UD `# newpar_block` comment under the `# newpar` comment, which may however feature nested tags, for example:

```CoNLL-U
# newpar
# newpar_block = list type:::"unordered" (10 s) | item (4 s)
```

This comment indicates the opening of a `<list type="unordered">` block element, which spans 10 sentences (`(10 s)`). However, the list begins with a nested block, a list item (i.e. a bullet point), which spans 4 sentences, as indicated after the pipe separator. For documentation of XML elements in GENTLE, please see the [GUM wiki](https://wiki.gucorpling.org/gum/tei_markup_in_gum).

More information and additional annotation layers can also be found in the GENTLE [source repo](https://github.com/gucorpling/gentle/).

# Metadata

Document metadata is given at the beginning of each new document in key-value pair comments beginning with the prefix `meta::`, as in:

```
# global.Entity = GRP-etype-infstat-centering-minspan-link-identity
# meta::dateCollected = 2019-09-10
# meta::dateCreated = 2004-08-14
# meta::dateModified = 2019-09-11
# meta::shortTitle = padalecki
# meta::sourceURL = https://en.wikipedia.org/wiki/Jared_Padalecki
# meta::speakerCount = 0
# meta::title = Jared Padalecki
```

# Documents and splits - test only

The entire corpus is designed to be a *test set* of challenging genres for NLP systems to be evaluated on. Although one can train a model on this corpus, or concatenate it to another training set, we present this entire corpus as a test set, and do not provide any official train / dev data.


# Acknowledgments

...

## References

...


# Changelog


* 2023-05-15 v2.12
  * Initial release in Universal Dependencies.


<pre>
=== Machine-readable metadata (DO NOT REMOVE!) ================================
Data available since: UD v2.12
License: CC BY-NC-SA 4.0
Includes text: yes
Genre: academic grammar-examples legal medical nonfiction poetry social spoken
Lemmas: manual native
UPOS: converted from manual
XPOS: manual native
Features: converted from manual
Relations: manual native
Contributors: Aoyama, Tatsuya; Behzad, Shabnam; Gessler, Luke; Levine, Lauren; Lin, Yi-Ju Jessica; Liu, Yang Janet; Peng, Siyao Logan; Zhu, Yilun; Zeldes, Amir
Contributing: elsewhere
Contact: amir.zeldes@georgetown.edu
===============================================================================
</pre>
