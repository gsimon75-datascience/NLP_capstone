TARGETS=NLP_wk3_Predictor_Model.html
#TARGETS=NLP_wk1_Milestone_Report.html
ID_WK2=https://api.rpubs.com/api/v1/document/361640/0b75a1ccca8b455e971a5dca3a883030

all:		$(TARGETS)

.PHONY:		clean upload_wk2
clean:
		rm -f $(TARGETS)

%.html %.md:	%.Rmd
		# echo 'library(knitr);knit2pdf("$^");' | R --no-save`
		echo 'library(rmarkdown); render("$^", output_format="html_document");' | R --no-save

upload_wk2:	$(TARGETS)
		echo 'library(markdown); result <- rpubsUpload("NLP Week 1 Milestone Report", "$<", "$(ID_WK2)"); print(result);' | R --no-save 
		
