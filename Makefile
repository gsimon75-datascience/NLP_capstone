TARGETS=NLP_wk1_Milestone_Report.html

all:		$(TARGETS)

.PHONY:		clean
clean:
		rm -f $(TARGETS)

%.html %.md:	%.Rmd
		# echo 'library(knitr);knit2pdf("$^");' | R --no-save`
		echo 'library(rmarkdown); render("$^", output_format="html_document");' | R --no-save

