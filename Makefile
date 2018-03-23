TARGETS=NLP_wk3_Predictor_Model.html
SUMMARY="NLP Week 3 Predictor Model"


#TARGETS=NLP_wk1_Milestone_Report.html
#SUMMARY="NLP Week 1 Milestone Report"
#ID=https://api.rpubs.com/api/v1/document/361640/0b75a1ccca8b455e971a5dca3a883030


all:		$(TARGETS)

.PHONY:		clean upload_wk2
clean:
		rm -f $(TARGETS)

%.html %.md:	%.Rmd
		# echo 'library(knitr);knit2pdf("$^");' | R --no-save`
		echo 'library(rmarkdown); render("$^", output_format="html_document");' | R --no-save

upload:		$(TARGETS)
		echo 'library(markdown); result <- rpubsUpload("$(SUMMARY}", "$<", "$(ID)"); print(result);' | R --no-save 
	


CC=g++
CPPFLAGS=-Wall -std=c++11 -O3
LDFLAGS=-lsqlite3

find_ngrams:	find_ngrams.cc 
