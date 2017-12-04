import * as Xmm from 'xmm-client';

const RAPID_MIX_DOC_VERSION = '1.0.0';
const RAPID_MIX_DEFAULT_LABEL = 'rapidmixDefaultLabel';

/**
 * @module rapidlib
 *
 * @description All the following functions convert from / to rapidMix / RapidLib JS JSON objects.
 */

/**
 * Convert a RapidMix training set Object to a RapidLib JS training set Object.
 *
 * @param {JSON} rapidMixTrainingSet - A RapidMix compatible training set
 *
 * @return {JSON} rapidLibTrainingSet - A RapidLib JS compatible training set
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

/**
 * @module xmm
 *
 * @description All the following functions convert from / to rapidMix / XMM JSON objects.
 */

/**
 * Convert a RapidMix training set Object to an XMM training set Object.
 *
 * @param {JSON} rapidMixTrainingSet - A RapidMix compatible training set
 *
 * @return {JSON} xmmTrainingSet - An XMM compatible training set
 */
const rapidMixToXmmTrainingSet = rapidMixTrainingSet => {
  const payload = rapidMixTrainingSet.payload;

  const config = {
    bimodal: payload.outputDimension > 0,
    dimension: payload.inputDimension + payload.outputDimension,
    dimensionInput: (payload.outputDimension > 0) ? payload.inputDimension : 0,
  };

  if (payload.columnNames) {
    config.columnNames = payload.columnNames.input.slice();
    config.columnNames = config.columnNames.concat(payload.columnNames.output);
  }

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
 * Convert an XMM training set Object to a RapidMix training set Object.
 *
 * @param {JSON} xmmTrainingSet - An XMM compatible training set
 *
 * @return {JSON} rapidMixTrainingSet - A RapidMix compatible training set
 */
const xmmToRapidMixTrainingSet = xmmTrainingSet => {
  const payload = {
    columnNames: { input: [], output: [] },
    data: []
  };
  const phrases = xmmTrainingSet.phrases;

  if (xmmTrainingSet.bimodal) {
    payload.inputDimension = xmmTrainingSet.dimension_input;
    payload.outputDimension = xmmTrainingSet.dimension - xmmTrainingSet.dimension_input;

    const iDim = payload.inputDimension;
    const oDim = payload.outputDimension;

    for (let i = 0; i < xmmTrainingSet.column_names.length; i++) {
      if (i < iDim) {
        payload.columnNames.input.push(xmmTrainingSet.column_names[i]);
      } else {
        payload.columnNames.output.push(xmmTrainingSet.column_names[i]);
      }
    }

    for (let i = 0; i < phrases.length; i++) {
      const example = {
        input:[],
        output: [],
        label: phrases[i].label
      };

      for (let j = 0; j < phrases[i].length; j++) {
        example.input.push(phrases[i].data_input.slice(j * iDim, (j + 1) * iDim));
        example.output.push(phrases[i].data_output.slice(j * oDim, (j + 1) * oDim));
      }

      payload.data.push(example);
    }
  } else {
    payload.inputDimension = xmmTrainingSet.dimension;
    payload.outputDimension = 0;

    const dim = payload.inputDimension;

    for (let i = 0; i < xmmTrainingSet.column_names.length; i++) {
      payload.columnNames.input.push(xmmTrainingSet.column_names[i]);
    }

    for (let i = 0; i < phrases.length; i++) {
      const example = {
        input:[],
        output: [],
        label: phrases[i].label
      };

      for (let j = 0; j < phrases[i].length; j++) {
        example.input.push(phrases[i].data.slice(j * dim, (j + 1) * dim));
      }

      payload.data.push(example);
    }
  }

  return {
    docType: 'rapid-mix:ml-training-set',
    docVersion: RAPID_MIX_DOC_VERSION,
    payload: payload,
  };
}

/**
 * Convert a RapidMix configuration Object to an XMM configuration Object.
 *
 * @param {JSON} rapidMixConfig - A RapidMix compatible configuraiton object
 *
 * @return {JSON} xmmConfig - A configuration object ready to be used by the XMM library
 */
const rapidMixToXmmConfig = rapidMixConfig => {
  return rapidMixConfig.payload;
}

/**
 * Convert an XMM configuration Object to a RapidMix configuration set Object.
 *
 * @param {JSON} xmmConfig - A configuration object targeting the XMM library
 *
 * @return {JSON} rapidMixConfig - A RapidMix compatible configuration object
 */
const xmmToRapidMixConfig = xmmConfig => {
  return {
    docType: 'rapid-mix:ml-configuration',
    docVersion: RAPID_MIX_DOC_VERSION,
    target: {
      name: `xmm:${xmmConfig.modelType}`,
      version: '1.0.0',
    },
    payload: xmmConfig,
  }
}

/**
 * Convert a RapidMix configuration Object to an XMM configuration Object.
 *
 * @param {JSON} rapidMixModel - A RapidMix compatible model
 *
 * @return {JSON} xmmModel - A model ready to be used by the XMM library
 */
const rapidMixToXmmModel = rapidMixModel => {
  return rapidMixModel.payload;
}

/**
 * Convert an XMM model Object to a RapidMix model Object.
 *
 * @param {JSON} xmmModel - A model generated by the XMM library
 *
 * @return {JSON} rapidMixModel - A RapidMix compatible model
 */
const xmmToRapidMixModel = xmmModel => {
  const modelType = xmmModel.configuration.default_parameters.states ? 'hhmm' : 'gmm';

  return {
    docType: 'rapid-mix:ml-model',
    docVersion: RAPID_MIX_DOC_VERSION,
    target: {
      name: `xmm:${modelType}`,
      version: '1.0.0',
    },
    payload: xmmModel,
  }
};

/**
 * @module como
 *
 * @description For the moment the como web service will only return XMM models
 * wrapped into RAPID-MIX JSON objects, taking RAPID-MIX trainings sets and XMM configurations.
 */

/**
 * Create the JSON to send to the Como web service via http request.
 *
 * @param {JSON} config - A valid RapidMix configuration object
 * @param {JSON} trainingSet - A valid RapidMix training set object
 * @param {JSON} [metas=null] - Some optional meta data
 * @param {JSON} [signalProcessing=null] - An optional description of the pre processing used to obtain the training set
 *
 * @return {JSON} httpRequest - A valid JSON to be sent to the Como web service via http request.
 */
const createComoHttpRequest = (config, trainingSet, metas = null, signalProcessing = null) => {
  const resquest = {
    docType: 'rapid-mix:ml-http-request',
    docVersion: RAPID_MIX_DOC_VERSION,
    target: {
      name: 'como-web-service',
      version: '1.0.0'
    },
    payload: {
      configuration: config,
      trainingSet: trainingSet
    }
  };

  if (metas !== null) {
    resquest.payload.metas = metas;
  }

  if (signalProcessing !== null) {
    resquest.payload.signalProcessing = signalProcessing;
  }

  return resquest;
};

/**
 * Create the JSON to send back as a response to http requests to the Como web service.
 *
 * @param {JSON} config - A valid RapidMix configuration object
 * @param {JSON} model - A valid RapidMix model object
 * @param {JSON} [metas=null] - Some optional meta data
 * @param {JSON} [signalProcessing=null] - An optional description of the pre processing used to obtain the training set
 *
 * @return {JSON} httpResponse - A valid JSON response to be sent back from the Como web service via http.
 */
const createComoHttpResponse = (config, model, metas = null, signalProcessing = null) => {
  const response = {
    docType: 'rapid-mix:ml-http-response',
    docVersion: RAPID_MIX_DOC_VERSION,
    target: {
      name: 'como-web-service',
      version: '1.0.0'
    },
    payload: {
      configuration: config,
      model: model
    }
  };

  if (metas !== null) {
    response.payload.metas = metas;
  }

  if (signalProcessing !== null) {
    response.payload.signalProcessing = signalProcessing;
  }

  return response;
};


export default {
  // rapidLib adapters
  rapidMixToRapidLibTrainingSet,

  // xmm adapters
  rapidMixToXmmTrainingSet,
  xmmToRapidMixTrainingSet,

  rapidMixToXmmConfig,
  xmmToRapidMixConfig,

  rapidMixToXmmModel,
  xmmToRapidMixModel,

  createComoHttpRequest,
  createComoHttpResponse,
  // constants
  RAPID_MIX_DOC_VERSION,
  RAPID_MIX_DEFAULT_LABEL
};
