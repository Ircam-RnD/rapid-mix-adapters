import * as Xmm from 'xmm-client';

const RAPID_MIX_DOC_VERSION = '1.0.0';
const RAPID_MIX_DEFAULT_LABEL = 'rapidmixDefaultLabel';

/**
 * Convert a RapidMix training set Object to an XMM training set Object.
 */
const rapidMixToXmmTrainingSet = rapidMixTrainingSet => {
  const payload = rapidMixTrainingSet.payload;

  const config = {
    bimodal: payload.outputDimension > 0,
    dimension: payload.inputDimension + payload.outputDimension,
    dimensionInput: (payload.outputDimension > 0) ? payload.inputDimension : 0,
  };

  const phraseMaker = new Xmm.PhraseMaker(config);
  const setMaker = new Xmm.SetMaker();

  for (let i = 0; i < payload.data.length; i++) {
    const datum = payload.data[i];

    phraseMaker.reset();
    phraseMaker.setConfig({ label: datum.label });

    for (let j = 0; j < datum.input.length; j++) {
      let vector = datum.input[j];

      if (payload.outputDimension > 0)
        vector = vector.concat(datum.output[j]);

      phraseMaker.addObservation(vector);
    }

    setMaker.addPhrase(phraseMaker.getPhrase());
  }

  return setMaker.getTrainingSet();
}

/**
 * Convert a RapidMix configuration Object to an XMM training set Object.
 */
const rapidMixToXmmConfig = rapidMixConfig => {
  return rapidMixConfig.payload;
}

/**
 * Convert an XMM model Object to a RapidMix model Object.
 */
const xmmToRapidMixModel = xmmModel => {
  const modelType = xmmModel.configuration.default_parameters.states ? 'hhmm' : 'gmm';

  return {
    docType: 'rapid-mix:ml:model',
    docVersion: RAPID_MIX_DOC_VERSION,
    target: {
      name: `xmm:${modelType}`,
      version: '1.0.0'
    },
    payload: xmmModel,
  }
};


/**
 * Convert a RapidMix training set Object to a RapidLib training set Object.
 */
const rapidMixToRapidLibTrainingSet = rapidMixTrainingSet => {
  const rapidLibTrainingSet = [];

  for (let i = 0; i < rapidMixTrainingSet.payload.data.length; i++) {
    const phrase = rapidMixTrainingSet.payload.data[i];

    for (let j = 0; j < phrase.input.length; j++) {
      const el = {
        label: phrase.label,
        input: phrase.input[j],
        output: phrase.output.length > 0 ? phrase.output[j] : [],
      };

      rapidLibTrainingSet.push(el);
    }
  }

  return rapidLibTrainingSet;
};


export default {
  // rapidLib adapters
  rapidMixToRapidLibTrainingSet,
  // xmm adapters
  rapidMixToXmmTrainingSet,
  xmmToRapidMixModel,

  // constants
  RAPID_MIX_DOC_VERSION,
  RAPID_MIX_DEFAULT_LABEL
};
