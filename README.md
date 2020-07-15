# SalesBot
Price inquiry bot (Java 8, Apache OpenNLP, WebSocket)

## Requirements
* Bot must be able to answer pricing and sales related questions in an intelligent manner
* It must be knowledgable enough to cover the most important questions in the mentioned field

## Installation
* Set up [Maven](https://maven.apache.org/download.cgi) and [JDK 11](https://adoptopenjdk.net/) on your machine
* Run `mvn clean install`
* Run `mvn package` to deploy a JAR file
* `java -jar /path/SalesBot.jar`
* Use a WebSocket client like [this](https://addons.mozilla.org/en/firefox/addon/simple-websocket-client/) to connect to the bot