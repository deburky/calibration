import { SSTConfig } from "sst";
import { Api, Stack } from "sst/constructs";

export default {
  config(_input) {
    return {
      name: "doinstruct-coding-challenge",
      region: "us-east-1",
    };
  },
  stacks(app) {
    app.stack(function Stack({ stack }) {
      const api = new Api(stack, "api", {
        routes: {
          "POST /generate-cards": "src/functions/generateCards.handler",
        },
      });

      stack.addOutputs({
        ApiEndpoint: api.url,
      });
    });
  },
} satisfies SSTConfig; 