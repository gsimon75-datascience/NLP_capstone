library(shiny)

shinyUI(
	fluidPage(
		tags$head(tags$script(HTML('
					Shiny.addCustomMessageHandler("jsCode",
						function(message) {
							console.log(message)
							eval(message.code);
						}
						);
					'))),
		titlePanel("Predictive Text Input"),
		br(),
		fluidRow(
			column(3, actionButton("guess1", label = "guess #1", width="100%")),
			column(3, actionButton("guess2", label = "guess #2", width="100%")),
			column(3, actionButton("guess3", label = "guess #3", width="100%")),
			column(3, actionButton("guess4", label = "guess #4", width="100%"))
		),
		hr(),
		fluidRow(
			column(2, h4("Your input:")),
			column(10, textInput("userinput", label=NULL, value="Enter text...", width="100%"))
		),
		fluidRow(
			column(1, offset=2, checkboxInput("interactive", label = "Interactive", value = FALSE)),
			column(1, offset=3, actionButton("generate", label = "Generate"))
		),
		conditionalPanel(condition="$('html').hasClass('shiny-busy')", style="text-align: center;", icon("spinner", "fa-3x fa-pulse")),
		hr()
	)
)
