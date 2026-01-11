* [ ] Add retry logic to GetCompletion
* [ ] Add retry logic if response does not match output schema
* [ ] Add Audio and Video output support
* [ ] Migrate RagSystem & WebSearch (possibly not needed)
* [ ] Add checks to see if the endpoint supports the methods called in "src/provider/completion.py" and also check if they return the same type of data if they do. If not throw an "Unsupported" Error.
* [ ] Add Sequential and Parallel Running Engines with data sharing between agents
* [ ] Evaluate usefuleness of toolcalling and implement tool-calling ability
* [ ] Same as above for MCP calling.
* [ ] The token counting might be broken (since whole response is being re-tokenized and cache is not being counted). Making it more thorough and robust might be useful.
