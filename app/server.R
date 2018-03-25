library(shiny)
library(readxl)
library(ggplot2)
library(lattice)
library(caret)
library(car)

setActionButtonDisabled <- function(id, session, state) {
	session$sendCustomMessage(type="jsCode", list(code= paste("$('#",id,"').prop('disabled',",state,")",sep="")))
}

rds_file <- "hstat.rds"

shinyServer(
	function(input, output, session) {
		#observe({
		observeEvent(input$interactive, {
			setActionButtonDisabled("generate", session, if (interactive()) "true" else "false")
		})

		interactive <- reactive({
			input$interactive
		})

		observeEvent({ if (input$interactive) input$userinput else 0}, {
			if (input$interactive) {
				g <- isolate(guess())
				updateActionButton(session, "guess1", label=g[1])
				updateActionButton(session, "guess2", label=g[2])
				updateActionButton(session, "guess3", label=g[3])
				updateActionButton(session, "guess4", label=g[4])
			}
		})

		#predictors <- reactive({
		#	g <- isolate(guess())
		#	c(input$userinput, input$guess1, input$guess2, input$guess3, input$guess4)
		#})

		observeEvent(input$guess1, {
			g <- isolate(guess())
			updateTextInput(session, "userinput", value=paste(isolate(input$userinput), g[1])) 
		})

		guess <- reactive({
			c(input$userinput, paste(input$userinput, "yadda"), "boo", "")
		})
	}
)

