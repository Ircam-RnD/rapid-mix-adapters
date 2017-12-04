'use strict';Object.defineProperty(exports,"__esModule",{value:true});var _xmmClient=require('xmm-client');var Xmm=_interopRequireWildcard(_xmmClient);function _interopRequireWildcard(obj){if(obj&&obj.__esModule){return obj;}else{var newObj={};if(obj!=null){for(var key in obj){if(Object.prototype.hasOwnProperty.call(obj,key))newObj[key]=obj[key];}}newObj.default=obj;return newObj;}}/*
 * @module constants
 *
 * @description Constants used by the RAPID-MIX JSON specification.
 *//**
 * @constant
 * @type {String}
 * @description The RAPID-MIX JSON document specification version.
 * @default
 */var RAPID_MIX_DOC_VERSION='1.0.0';/**
 * @constant
 * @type {String}
 * @description The default RAPID-MIX label used to build training sets.
 * @default
 */var RAPID_MIX_DEFAULT_LABEL='rapidmixDefaultLabel';/*
 * @module rapidlib
 *
 * @description All the following functions convert from / to RAPID-MIX / RapidLib JS JSON objects.
 *//**
 * Convert a RAPID-MIX training set Object to a RapidLib JS training set Object.
 *
 * @param {JSON} rapidMixTrainingSet - A RAPID-MIX compatible training set
 *
 * @return {JSON} rapidLibTrainingSet - A RapidLib JS compatible training set
 */var rapidMixToRapidLibTrainingSet=function rapidMixToRapidLibTrainingSet(rapidMixTrainingSet){var rapidLibTrainingSet=[];for(var i=0;i<rapidMixTrainingSet.payload.data.length;i++){var phrase=rapidMixTrainingSet.payload.data[i];for(var j=0;j<phrase.input.length;j++){var el={label:phrase.label,input:phrase.input[j],output:phrase.output.length>0?phrase.output[j]:[]};rapidLibTrainingSet.push(el);}}return rapidLibTrainingSet;};/*
 * @module xmm
 *
 * @description All the following functions convert from / to rapidMix / XMM JSON objects.
 *//**
 * Convert a RAPID-MIX training set Object to an XMM training set Object.
 *
 * @param {JSON} rapidMixTrainingSet - A RAPID-MIX compatible training set
 *
 * @return {JSON} xmmTrainingSet - An XMM compatible training set
 */var rapidMixToXmmTrainingSet=function rapidMixToXmmTrainingSet(rapidMixTrainingSet){var payload=rapidMixTrainingSet.payload;var config={bimodal:payload.outputDimension>0,dimension:payload.inputDimension+payload.outputDimension,dimensionInput:payload.outputDimension>0?payload.inputDimension:0};if(payload.columnNames){config.columnNames=payload.columnNames.input.slice();config.columnNames=config.columnNames.concat(payload.columnNames.output);}var phraseMaker=new Xmm.PhraseMaker(config);var setMaker=new Xmm.SetMaker();for(var i=0;i<payload.data.length;i++){var datum=payload.data[i];phraseMaker.reset();phraseMaker.setConfig({label:datum.label});for(var j=0;j<datum.input.length;j++){var vector=datum.input[j];if(payload.outputDimension>0)vector=vector.concat(datum.output[j]);phraseMaker.addObservation(vector);}setMaker.addPhrase(phraseMaker.getPhrase());}return setMaker.getTrainingSet();};/**
 * Convert an XMM training set Object to a RAPID-MIX training set Object.
 *
 * @param {JSON} xmmTrainingSet - An XMM compatible training set
 *
 * @return {JSON} rapidMixTrainingSet - A RAPID-MIX compatible training set
 */var xmmToRapidMixTrainingSet=function xmmToRapidMixTrainingSet(xmmTrainingSet){var payload={columnNames:{input:[],output:[]},data:[]};var phrases=xmmTrainingSet.phrases;if(xmmTrainingSet.bimodal){payload.inputDimension=xmmTrainingSet.dimension_input;payload.outputDimension=xmmTrainingSet.dimension-xmmTrainingSet.dimension_input;var iDim=payload.inputDimension;var oDim=payload.outputDimension;for(var i=0;i<xmmTrainingSet.column_names.length;i++){if(i<iDim){payload.columnNames.input.push(xmmTrainingSet.column_names[i]);}else{payload.columnNames.output.push(xmmTrainingSet.column_names[i]);}}for(var _i=0;_i<phrases.length;_i++){var example={input:[],output:[],label:phrases[_i].label};for(var j=0;j<phrases[_i].length;j++){example.input.push(phrases[_i].data_input.slice(j*iDim,(j+1)*iDim));example.output.push(phrases[_i].data_output.slice(j*oDim,(j+1)*oDim));}payload.data.push(example);}}else{payload.inputDimension=xmmTrainingSet.dimension;payload.outputDimension=0;var dim=payload.inputDimension;for(var _i2=0;_i2<xmmTrainingSet.column_names.length;_i2++){payload.columnNames.input.push(xmmTrainingSet.column_names[_i2]);}for(var _i3=0;_i3<phrases.length;_i3++){var _example={input:[],output:[],label:phrases[_i3].label};for(var _j=0;_j<phrases[_i3].length;_j++){_example.input.push(phrases[_i3].data.slice(_j*dim,(_j+1)*dim));}payload.data.push(_example);}}return{docType:'rapid-mix:ml-training-set',docVersion:RAPID_MIX_DOC_VERSION,payload:payload};};/**
 * Convert a RAPID-MIX configuration Object to an XMM configuration Object.
 *
 * @param {JSON} rapidMixConfig - A RAPID-MIX compatible configuraiton object
 *
 * @return {JSON} xmmConfig - A configuration object ready to be used by the XMM library
 */var rapidMixToXmmConfig=function rapidMixToXmmConfig(rapidMixConfig){return rapidMixConfig.payload;};/**
 * Convert an XMM configuration Object to a RAPID-MIX configuration set Object.
 *
 * @param {JSON} xmmConfig - A configuration object targeting the XMM library
 *
 * @return {JSON} rapidMixConfig - A RAPID-MIX compatible configuration object
 */var xmmToRapidMixConfig=function xmmToRapidMixConfig(xmmConfig){return{docType:'rapid-mix:ml-configuration',docVersion:RAPID_MIX_DOC_VERSION,target:{name:'xmm:'+xmmConfig.modelType,version:'1.0.0'},payload:xmmConfig};};/**
 * Convert a RAPID-MIX configuration Object to an XMM configuration Object.
 *
 * @param {JSON} rapidMixModel - A RAPID-MIX compatible model
 *
 * @return {JSON} xmmModel - A model ready to be used by the XMM library
 */var rapidMixToXmmModel=function rapidMixToXmmModel(rapidMixModel){return rapidMixModel.payload;};/**
 * Convert an XMM model Object to a RAPID-MIX model Object.
 *
 * @param {JSON} xmmModel - A model generated by the XMM library
 *
 * @return {JSON} rapidMixModel - A RAPID-MIX compatible model
 */var xmmToRapidMixModel=function xmmToRapidMixModel(xmmModel){var modelType=xmmModel.configuration.default_parameters.states?'hhmm':'gmm';return{docType:'rapid-mix:ml-model',docVersion:RAPID_MIX_DOC_VERSION,target:{name:'xmm:'+modelType,version:'1.0.0'},payload:xmmModel};};/*
 * @module como
 *
 * @description For the moment the como web service will only return XMM models
 * wrapped into RAPID-MIX JSON objects, taking RAPID-MIX trainings sets and XMM configurations.
 *//**
 * Create the JSON to send to the Como web service via http request.
 *
 * @param {JSON} config - A valid RAPID-MIX configuration object
 * @param {JSON} trainingSet - A valid RAPID-MIX training set object
 * @param {JSON} [metas=null] - Some optional meta data
 * @param {JSON} [signalProcessing=null] - An optional description of the pre processing used to obtain the training set
 *
 * @return {JSON} httpRequest - A valid JSON to be sent to the Como web service via http request.
 */var createComoHttpRequest=function createComoHttpRequest(config,trainingSet){var metas=arguments.length>2&&arguments[2]!==undefined?arguments[2]:null;var signalProcessing=arguments.length>3&&arguments[3]!==undefined?arguments[3]:null;var resquest={docType:'rapid-mix:ml-http-request',docVersion:RAPID_MIX_DOC_VERSION,target:{name:'como-web-service',version:'1.0.0'},payload:{configuration:config,trainingSet:trainingSet}};if(metas!==null){resquest.payload.metas=metas;}if(signalProcessing!==null){resquest.payload.signalProcessing=signalProcessing;}return resquest;};/**
 * Create the JSON to send back as a response to http requests to the Como web service.
 *
 * @param {JSON} config - A valid RAPID-MIX configuration object
 * @param {JSON} model - A valid RAPID-MIX model object
 * @param {JSON} [metas=null] - Some optional meta data
 * @param {JSON} [signalProcessing=null] - An optional description of the pre processing used to obtain the training set
 *
 * @return {JSON} httpResponse - A valid JSON response to be sent back from the Como web service via http.
 */var createComoHttpResponse=function createComoHttpResponse(config,model){var metas=arguments.length>2&&arguments[2]!==undefined?arguments[2]:null;var signalProcessing=arguments.length>3&&arguments[3]!==undefined?arguments[3]:null;var response={docType:'rapid-mix:ml-http-response',docVersion:RAPID_MIX_DOC_VERSION,target:{name:'como-web-service',version:'1.0.0'},payload:{configuration:config,model:model}};if(metas!==null){response.payload.metas=metas;}if(signalProcessing!==null){response.payload.signalProcessing=signalProcessing;}return response;};exports.default={// rapidLib adapters
rapidMixToRapidLibTrainingSet:rapidMixToRapidLibTrainingSet,// xmm adapters
rapidMixToXmmTrainingSet:rapidMixToXmmTrainingSet,xmmToRapidMixTrainingSet:xmmToRapidMixTrainingSet,rapidMixToXmmConfig:rapidMixToXmmConfig,xmmToRapidMixConfig:xmmToRapidMixConfig,rapidMixToXmmModel:rapidMixToXmmModel,xmmToRapidMixModel:xmmToRapidMixModel,createComoHttpRequest:createComoHttpRequest,createComoHttpResponse:createComoHttpResponse,// constants
RAPID_MIX_DOC_VERSION:RAPID_MIX_DOC_VERSION,RAPID_MIX_DEFAULT_LABEL:RAPID_MIX_DEFAULT_LABEL};
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImluZGV4LmpzIl0sIm5hbWVzIjpbIlhtbSIsIlJBUElEX01JWF9ET0NfVkVSU0lPTiIsIlJBUElEX01JWF9ERUZBVUxUX0xBQkVMIiwicmFwaWRNaXhUb1JhcGlkTGliVHJhaW5pbmdTZXQiLCJyYXBpZExpYlRyYWluaW5nU2V0IiwiaSIsInJhcGlkTWl4VHJhaW5pbmdTZXQiLCJwYXlsb2FkIiwiZGF0YSIsImxlbmd0aCIsInBocmFzZSIsImoiLCJpbnB1dCIsImVsIiwibGFiZWwiLCJvdXRwdXQiLCJwdXNoIiwicmFwaWRNaXhUb1htbVRyYWluaW5nU2V0IiwiY29uZmlnIiwiYmltb2RhbCIsIm91dHB1dERpbWVuc2lvbiIsImRpbWVuc2lvbiIsImlucHV0RGltZW5zaW9uIiwiZGltZW5zaW9uSW5wdXQiLCJjb2x1bW5OYW1lcyIsInNsaWNlIiwiY29uY2F0IiwicGhyYXNlTWFrZXIiLCJQaHJhc2VNYWtlciIsInNldE1ha2VyIiwiU2V0TWFrZXIiLCJkYXR1bSIsInJlc2V0Iiwic2V0Q29uZmlnIiwidmVjdG9yIiwiYWRkT2JzZXJ2YXRpb24iLCJhZGRQaHJhc2UiLCJnZXRQaHJhc2UiLCJnZXRUcmFpbmluZ1NldCIsInhtbVRvUmFwaWRNaXhUcmFpbmluZ1NldCIsInBocmFzZXMiLCJ4bW1UcmFpbmluZ1NldCIsImRpbWVuc2lvbl9pbnB1dCIsImlEaW0iLCJvRGltIiwiY29sdW1uX25hbWVzIiwiZXhhbXBsZSIsImRhdGFfaW5wdXQiLCJkYXRhX291dHB1dCIsImRpbSIsImRvY1R5cGUiLCJkb2NWZXJzaW9uIiwicmFwaWRNaXhUb1htbUNvbmZpZyIsInJhcGlkTWl4Q29uZmlnIiwieG1tVG9SYXBpZE1peENvbmZpZyIsInRhcmdldCIsIm5hbWUiLCJ4bW1Db25maWciLCJtb2RlbFR5cGUiLCJ2ZXJzaW9uIiwicmFwaWRNaXhUb1htbU1vZGVsIiwicmFwaWRNaXhNb2RlbCIsInhtbVRvUmFwaWRNaXhNb2RlbCIsInhtbU1vZGVsIiwiY29uZmlndXJhdGlvbiIsImRlZmF1bHRfcGFyYW1ldGVycyIsInN0YXRlcyIsImNyZWF0ZUNvbW9IdHRwUmVxdWVzdCIsInRyYWluaW5nU2V0IiwibWV0YXMiLCJzaWduYWxQcm9jZXNzaW5nIiwicmVzcXVlc3QiLCJjcmVhdGVDb21vSHR0cFJlc3BvbnNlIiwibW9kZWwiLCJyZXNwb25zZSJdLCJtYXBwaW5ncyI6InNFQUFBLHFDLEdBQVlBLEksK1FBRVo7Ozs7R0FNQTs7Ozs7R0FNQSxHQUFNQyx1QkFBd0IsT0FBOUIsQ0FFQTs7Ozs7R0FNQSxHQUFNQyx5QkFBMEIsc0JBQWhDLENBRUE7Ozs7R0FNQTs7Ozs7O0dBT0EsR0FBTUMsK0JBQWdDLFFBQWhDQSw4QkFBZ0MscUJBQXVCLENBQzNELEdBQU1DLHFCQUFzQixFQUE1QixDQUVBLElBQUssR0FBSUMsR0FBSSxDQUFiLENBQWdCQSxFQUFJQyxvQkFBb0JDLE9BQXBCLENBQTRCQyxJQUE1QixDQUFpQ0MsTUFBckQsQ0FBNkRKLEdBQTdELENBQWtFLENBQ2hFLEdBQU1LLFFBQVNKLG9CQUFvQkMsT0FBcEIsQ0FBNEJDLElBQTVCLENBQWlDSCxDQUFqQyxDQUFmLENBRUEsSUFBSyxHQUFJTSxHQUFJLENBQWIsQ0FBZ0JBLEVBQUlELE9BQU9FLEtBQVAsQ0FBYUgsTUFBakMsQ0FBeUNFLEdBQXpDLENBQThDLENBQzVDLEdBQU1FLElBQUssQ0FDVEMsTUFBT0osT0FBT0ksS0FETCxDQUVURixNQUFPRixPQUFPRSxLQUFQLENBQWFELENBQWIsQ0FGRSxDQUdUSSxPQUFRTCxPQUFPSyxNQUFQLENBQWNOLE1BQWQsQ0FBdUIsQ0FBdkIsQ0FBMkJDLE9BQU9LLE1BQVAsQ0FBY0osQ0FBZCxDQUEzQixDQUE4QyxFQUg3QyxDQUFYLENBTUFQLG9CQUFvQlksSUFBcEIsQ0FBeUJILEVBQXpCLEVBQ0QsQ0FDRixDQUVELE1BQU9ULG9CQUFQLENBQ0QsQ0FsQkQsQ0FvQkE7Ozs7R0FNQTs7Ozs7O0dBT0EsR0FBTWEsMEJBQTJCLFFBQTNCQSx5QkFBMkIscUJBQXVCLENBQ3RELEdBQU1WLFNBQVVELG9CQUFvQkMsT0FBcEMsQ0FFQSxHQUFNVyxRQUFTLENBQ2JDLFFBQVNaLFFBQVFhLGVBQVIsQ0FBMEIsQ0FEdEIsQ0FFYkMsVUFBV2QsUUFBUWUsY0FBUixDQUF5QmYsUUFBUWEsZUFGL0IsQ0FHYkcsZUFBaUJoQixRQUFRYSxlQUFSLENBQTBCLENBQTNCLENBQWdDYixRQUFRZSxjQUF4QyxDQUF5RCxDQUg1RCxDQUFmLENBTUEsR0FBSWYsUUFBUWlCLFdBQVosQ0FBeUIsQ0FDdkJOLE9BQU9NLFdBQVAsQ0FBcUJqQixRQUFRaUIsV0FBUixDQUFvQlosS0FBcEIsQ0FBMEJhLEtBQTFCLEVBQXJCLENBQ0FQLE9BQU9NLFdBQVAsQ0FBcUJOLE9BQU9NLFdBQVAsQ0FBbUJFLE1BQW5CLENBQTBCbkIsUUFBUWlCLFdBQVIsQ0FBb0JULE1BQTlDLENBQXJCLENBQ0QsQ0FFRCxHQUFNWSxhQUFjLEdBQUkzQixLQUFJNEIsV0FBUixDQUFvQlYsTUFBcEIsQ0FBcEIsQ0FDQSxHQUFNVyxVQUFXLEdBQUk3QixLQUFJOEIsUUFBUixFQUFqQixDQUVBLElBQUssR0FBSXpCLEdBQUksQ0FBYixDQUFnQkEsRUFBSUUsUUFBUUMsSUFBUixDQUFhQyxNQUFqQyxDQUF5Q0osR0FBekMsQ0FBOEMsQ0FDNUMsR0FBTTBCLE9BQVF4QixRQUFRQyxJQUFSLENBQWFILENBQWIsQ0FBZCxDQUVBc0IsWUFBWUssS0FBWixHQUNBTCxZQUFZTSxTQUFaLENBQXNCLENBQUVuQixNQUFPaUIsTUFBTWpCLEtBQWYsQ0FBdEIsRUFFQSxJQUFLLEdBQUlILEdBQUksQ0FBYixDQUFnQkEsRUFBSW9CLE1BQU1uQixLQUFOLENBQVlILE1BQWhDLENBQXdDRSxHQUF4QyxDQUE2QyxDQUMzQyxHQUFJdUIsUUFBU0gsTUFBTW5CLEtBQU4sQ0FBWUQsQ0FBWixDQUFiLENBRUEsR0FBSUosUUFBUWEsZUFBUixDQUEwQixDQUE5QixDQUNFYyxPQUFTQSxPQUFPUixNQUFQLENBQWNLLE1BQU1oQixNQUFOLENBQWFKLENBQWIsQ0FBZCxDQUFULENBRUZnQixZQUFZUSxjQUFaLENBQTJCRCxNQUEzQixFQUNELENBRURMLFNBQVNPLFNBQVQsQ0FBbUJULFlBQVlVLFNBQVosRUFBbkIsRUFDRCxDQUVELE1BQU9SLFVBQVNTLGNBQVQsRUFBUCxDQUNELENBcENELENBc0NBOzs7Ozs7R0FPQSxHQUFNQywwQkFBMkIsUUFBM0JBLHlCQUEyQixnQkFBa0IsQ0FDakQsR0FBTWhDLFNBQVUsQ0FDZGlCLFlBQWEsQ0FBRVosTUFBTyxFQUFULENBQWFHLE9BQVEsRUFBckIsQ0FEQyxDQUVkUCxLQUFNLEVBRlEsQ0FBaEIsQ0FJQSxHQUFNZ0MsU0FBVUMsZUFBZUQsT0FBL0IsQ0FFQSxHQUFJQyxlQUFldEIsT0FBbkIsQ0FBNEIsQ0FDMUJaLFFBQVFlLGNBQVIsQ0FBeUJtQixlQUFlQyxlQUF4QyxDQUNBbkMsUUFBUWEsZUFBUixDQUEwQnFCLGVBQWVwQixTQUFmLENBQTJCb0IsZUFBZUMsZUFBcEUsQ0FFQSxHQUFNQyxNQUFPcEMsUUFBUWUsY0FBckIsQ0FDQSxHQUFNc0IsTUFBT3JDLFFBQVFhLGVBQXJCLENBRUEsSUFBSyxHQUFJZixHQUFJLENBQWIsQ0FBZ0JBLEVBQUlvQyxlQUFlSSxZQUFmLENBQTRCcEMsTUFBaEQsQ0FBd0RKLEdBQXhELENBQTZELENBQzNELEdBQUlBLEVBQUlzQyxJQUFSLENBQWMsQ0FDWnBDLFFBQVFpQixXQUFSLENBQW9CWixLQUFwQixDQUEwQkksSUFBMUIsQ0FBK0J5QixlQUFlSSxZQUFmLENBQTRCeEMsQ0FBNUIsQ0FBL0IsRUFDRCxDQUZELElBRU8sQ0FDTEUsUUFBUWlCLFdBQVIsQ0FBb0JULE1BQXBCLENBQTJCQyxJQUEzQixDQUFnQ3lCLGVBQWVJLFlBQWYsQ0FBNEJ4QyxDQUE1QixDQUFoQyxFQUNELENBQ0YsQ0FFRCxJQUFLLEdBQUlBLElBQUksQ0FBYixDQUFnQkEsR0FBSW1DLFFBQVEvQixNQUE1QixDQUFvQ0osSUFBcEMsQ0FBeUMsQ0FDdkMsR0FBTXlDLFNBQVUsQ0FDZGxDLE1BQU0sRUFEUSxDQUVkRyxPQUFRLEVBRk0sQ0FHZEQsTUFBTzBCLFFBQVFuQyxFQUFSLEVBQVdTLEtBSEosQ0FBaEIsQ0FNQSxJQUFLLEdBQUlILEdBQUksQ0FBYixDQUFnQkEsRUFBSTZCLFFBQVFuQyxFQUFSLEVBQVdJLE1BQS9CLENBQXVDRSxHQUF2QyxDQUE0QyxDQUMxQ21DLFFBQVFsQyxLQUFSLENBQWNJLElBQWQsQ0FBbUJ3QixRQUFRbkMsRUFBUixFQUFXMEMsVUFBWCxDQUFzQnRCLEtBQXRCLENBQTRCZCxFQUFJZ0MsSUFBaEMsQ0FBc0MsQ0FBQ2hDLEVBQUksQ0FBTCxFQUFVZ0MsSUFBaEQsQ0FBbkIsRUFDQUcsUUFBUS9CLE1BQVIsQ0FBZUMsSUFBZixDQUFvQndCLFFBQVFuQyxFQUFSLEVBQVcyQyxXQUFYLENBQXVCdkIsS0FBdkIsQ0FBNkJkLEVBQUlpQyxJQUFqQyxDQUF1QyxDQUFDakMsRUFBSSxDQUFMLEVBQVVpQyxJQUFqRCxDQUFwQixFQUNELENBRURyQyxRQUFRQyxJQUFSLENBQWFRLElBQWIsQ0FBa0I4QixPQUFsQixFQUNELENBQ0YsQ0E3QkQsSUE2Qk8sQ0FDTHZDLFFBQVFlLGNBQVIsQ0FBeUJtQixlQUFlcEIsU0FBeEMsQ0FDQWQsUUFBUWEsZUFBUixDQUEwQixDQUExQixDQUVBLEdBQU02QixLQUFNMUMsUUFBUWUsY0FBcEIsQ0FFQSxJQUFLLEdBQUlqQixLQUFJLENBQWIsQ0FBZ0JBLElBQUlvQyxlQUFlSSxZQUFmLENBQTRCcEMsTUFBaEQsQ0FBd0RKLEtBQXhELENBQTZELENBQzNERSxRQUFRaUIsV0FBUixDQUFvQlosS0FBcEIsQ0FBMEJJLElBQTFCLENBQStCeUIsZUFBZUksWUFBZixDQUE0QnhDLEdBQTVCLENBQS9CLEVBQ0QsQ0FFRCxJQUFLLEdBQUlBLEtBQUksQ0FBYixDQUFnQkEsSUFBSW1DLFFBQVEvQixNQUE1QixDQUFvQ0osS0FBcEMsQ0FBeUMsQ0FDdkMsR0FBTXlDLFVBQVUsQ0FDZGxDLE1BQU0sRUFEUSxDQUVkRyxPQUFRLEVBRk0sQ0FHZEQsTUFBTzBCLFFBQVFuQyxHQUFSLEVBQVdTLEtBSEosQ0FBaEIsQ0FNQSxJQUFLLEdBQUlILElBQUksQ0FBYixDQUFnQkEsR0FBSTZCLFFBQVFuQyxHQUFSLEVBQVdJLE1BQS9CLENBQXVDRSxJQUF2QyxDQUE0QyxDQUMxQ21DLFNBQVFsQyxLQUFSLENBQWNJLElBQWQsQ0FBbUJ3QixRQUFRbkMsR0FBUixFQUFXRyxJQUFYLENBQWdCaUIsS0FBaEIsQ0FBc0JkLEdBQUlzQyxHQUExQixDQUErQixDQUFDdEMsR0FBSSxDQUFMLEVBQVVzQyxHQUF6QyxDQUFuQixFQUNELENBRUQxQyxRQUFRQyxJQUFSLENBQWFRLElBQWIsQ0FBa0I4QixRQUFsQixFQUNELENBQ0YsQ0FFRCxNQUFPLENBQ0xJLFFBQVMsMkJBREosQ0FFTEMsV0FBWWxELHFCQUZQLENBR0xNLFFBQVNBLE9BSEosQ0FBUCxDQUtELENBbEVELENBb0VBOzs7Ozs7R0FPQSxHQUFNNkMscUJBQXNCLFFBQXRCQSxvQkFBc0IsZ0JBQWtCLENBQzVDLE1BQU9DLGdCQUFlOUMsT0FBdEIsQ0FDRCxDQUZELENBSUE7Ozs7OztHQU9BLEdBQU0rQyxxQkFBc0IsUUFBdEJBLG9CQUFzQixXQUFhLENBQ3ZDLE1BQU8sQ0FDTEosUUFBUyw0QkFESixDQUVMQyxXQUFZbEQscUJBRlAsQ0FHTHNELE9BQVEsQ0FDTkMsWUFBYUMsVUFBVUMsU0FEakIsQ0FFTkMsUUFBUyxPQUZILENBSEgsQ0FPTHBELFFBQVNrRCxTQVBKLENBQVAsQ0FTRCxDQVZELENBWUE7Ozs7OztHQU9BLEdBQU1HLG9CQUFxQixRQUFyQkEsbUJBQXFCLGVBQWlCLENBQzFDLE1BQU9DLGVBQWN0RCxPQUFyQixDQUNELENBRkQsQ0FJQTs7Ozs7O0dBT0EsR0FBTXVELG9CQUFxQixRQUFyQkEsbUJBQXFCLFVBQVksQ0FDckMsR0FBTUosV0FBWUssU0FBU0MsYUFBVCxDQUF1QkMsa0JBQXZCLENBQTBDQyxNQUExQyxDQUFtRCxNQUFuRCxDQUE0RCxLQUE5RSxDQUVBLE1BQU8sQ0FDTGhCLFFBQVMsb0JBREosQ0FFTEMsV0FBWWxELHFCQUZQLENBR0xzRCxPQUFRLENBQ05DLFlBQWFFLFNBRFAsQ0FFTkMsUUFBUyxPQUZILENBSEgsQ0FPTHBELFFBQVN3RCxRQVBKLENBQVAsQ0FTRCxDQVpELENBY0E7Ozs7O0dBT0E7Ozs7Ozs7OztHQVVBLEdBQU1JLHVCQUF3QixRQUF4QkEsc0JBQXdCLENBQUNqRCxNQUFELENBQVNrRCxXQUFULENBQWdFLElBQTFDQyxNQUEwQywyREFBbEMsSUFBa0MsSUFBNUJDLGlCQUE0QiwyREFBVCxJQUFTLENBQzVGLEdBQU1DLFVBQVcsQ0FDZnJCLFFBQVMsMkJBRE0sQ0FFZkMsV0FBWWxELHFCQUZHLENBR2ZzRCxPQUFRLENBQ05DLEtBQU0sa0JBREEsQ0FFTkcsUUFBUyxPQUZILENBSE8sQ0FPZnBELFFBQVMsQ0FDUHlELGNBQWU5QyxNQURSLENBRVBrRCxZQUFhQSxXQUZOLENBUE0sQ0FBakIsQ0FhQSxHQUFJQyxRQUFVLElBQWQsQ0FBb0IsQ0FDbEJFLFNBQVNoRSxPQUFULENBQWlCOEQsS0FBakIsQ0FBeUJBLEtBQXpCLENBQ0QsQ0FFRCxHQUFJQyxtQkFBcUIsSUFBekIsQ0FBK0IsQ0FDN0JDLFNBQVNoRSxPQUFULENBQWlCK0QsZ0JBQWpCLENBQW9DQSxnQkFBcEMsQ0FDRCxDQUVELE1BQU9DLFNBQVAsQ0FDRCxDQXZCRCxDQXlCQTs7Ozs7Ozs7O0dBVUEsR0FBTUMsd0JBQXlCLFFBQXpCQSx1QkFBeUIsQ0FBQ3RELE1BQUQsQ0FBU3VELEtBQVQsQ0FBMEQsSUFBMUNKLE1BQTBDLDJEQUFsQyxJQUFrQyxJQUE1QkMsaUJBQTRCLDJEQUFULElBQVMsQ0FDdkYsR0FBTUksVUFBVyxDQUNmeEIsUUFBUyw0QkFETSxDQUVmQyxXQUFZbEQscUJBRkcsQ0FHZnNELE9BQVEsQ0FDTkMsS0FBTSxrQkFEQSxDQUVORyxRQUFTLE9BRkgsQ0FITyxDQU9mcEQsUUFBUyxDQUNQeUQsY0FBZTlDLE1BRFIsQ0FFUHVELE1BQU9BLEtBRkEsQ0FQTSxDQUFqQixDQWFBLEdBQUlKLFFBQVUsSUFBZCxDQUFvQixDQUNsQkssU0FBU25FLE9BQVQsQ0FBaUI4RCxLQUFqQixDQUF5QkEsS0FBekIsQ0FDRCxDQUVELEdBQUlDLG1CQUFxQixJQUF6QixDQUErQixDQUM3QkksU0FBU25FLE9BQVQsQ0FBaUIrRCxnQkFBakIsQ0FBb0NBLGdCQUFwQyxDQUNELENBRUQsTUFBT0ksU0FBUCxDQUNELENBdkJELEMsZ0JBMEJlLENBQ2I7QUFDQXZFLDJEQUZhLENBSWI7QUFDQWMsaURBTGEsQ0FNYnNCLGlEQU5hLENBUWJhLHVDQVJhLENBU2JFLHVDQVRhLENBV2JNLHFDQVhhLENBWWJFLHFDQVphLENBY2JLLDJDQWRhLENBZWJLLDZDQWZhLENBZ0JiO0FBQ0F2RSwyQ0FqQmEsQ0FrQmJDLCtDQWxCYSxDIiwiZmlsZSI6ImluZGV4LmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0ICogYXMgWG1tIGZyb20gJ3htbS1jbGllbnQnO1xuXG4vKlxuICogQG1vZHVsZSBjb25zdGFudHNcbiAqXG4gKiBAZGVzY3JpcHRpb24gQ29uc3RhbnRzIHVzZWQgYnkgdGhlIFJBUElELU1JWCBKU09OIHNwZWNpZmljYXRpb24uXG4gKi9cblxuLyoqXG4gKiBAY29uc3RhbnRcbiAqIEB0eXBlIHtTdHJpbmd9XG4gKiBAZGVzY3JpcHRpb24gVGhlIFJBUElELU1JWCBKU09OIGRvY3VtZW50IHNwZWNpZmljYXRpb24gdmVyc2lvbi5cbiAqIEBkZWZhdWx0XG4gKi9cbmNvbnN0IFJBUElEX01JWF9ET0NfVkVSU0lPTiA9ICcxLjAuMCc7XG5cbi8qKlxuICogQGNvbnN0YW50XG4gKiBAdHlwZSB7U3RyaW5nfVxuICogQGRlc2NyaXB0aW9uIFRoZSBkZWZhdWx0IFJBUElELU1JWCBsYWJlbCB1c2VkIHRvIGJ1aWxkIHRyYWluaW5nIHNldHMuXG4gKiBAZGVmYXVsdFxuICovXG5jb25zdCBSQVBJRF9NSVhfREVGQVVMVF9MQUJFTCA9ICdyYXBpZG1peERlZmF1bHRMYWJlbCc7XG5cbi8qXG4gKiBAbW9kdWxlIHJhcGlkbGliXG4gKlxuICogQGRlc2NyaXB0aW9uIEFsbCB0aGUgZm9sbG93aW5nIGZ1bmN0aW9ucyBjb252ZXJ0IGZyb20gLyB0byBSQVBJRC1NSVggLyBSYXBpZExpYiBKUyBKU09OIG9iamVjdHMuXG4gKi9cblxuLyoqXG4gKiBDb252ZXJ0IGEgUkFQSUQtTUlYIHRyYWluaW5nIHNldCBPYmplY3QgdG8gYSBSYXBpZExpYiBKUyB0cmFpbmluZyBzZXQgT2JqZWN0LlxuICpcbiAqIEBwYXJhbSB7SlNPTn0gcmFwaWRNaXhUcmFpbmluZ1NldCAtIEEgUkFQSUQtTUlYIGNvbXBhdGlibGUgdHJhaW5pbmcgc2V0XG4gKlxuICogQHJldHVybiB7SlNPTn0gcmFwaWRMaWJUcmFpbmluZ1NldCAtIEEgUmFwaWRMaWIgSlMgY29tcGF0aWJsZSB0cmFpbmluZyBzZXRcbiAqL1xuY29uc3QgcmFwaWRNaXhUb1JhcGlkTGliVHJhaW5pbmdTZXQgPSByYXBpZE1peFRyYWluaW5nU2V0ID0+IHtcbiAgY29uc3QgcmFwaWRMaWJUcmFpbmluZ1NldCA9IFtdO1xuXG4gIGZvciAobGV0IGkgPSAwOyBpIDwgcmFwaWRNaXhUcmFpbmluZ1NldC5wYXlsb2FkLmRhdGEubGVuZ3RoOyBpKyspIHtcbiAgICBjb25zdCBwaHJhc2UgPSByYXBpZE1peFRyYWluaW5nU2V0LnBheWxvYWQuZGF0YVtpXTtcblxuICAgIGZvciAobGV0IGogPSAwOyBqIDwgcGhyYXNlLmlucHV0Lmxlbmd0aDsgaisrKSB7XG4gICAgICBjb25zdCBlbCA9IHtcbiAgICAgICAgbGFiZWw6IHBocmFzZS5sYWJlbCxcbiAgICAgICAgaW5wdXQ6IHBocmFzZS5pbnB1dFtqXSxcbiAgICAgICAgb3V0cHV0OiBwaHJhc2Uub3V0cHV0Lmxlbmd0aCA+IDAgPyBwaHJhc2Uub3V0cHV0W2pdIDogW10sXG4gICAgICB9O1xuXG4gICAgICByYXBpZExpYlRyYWluaW5nU2V0LnB1c2goZWwpO1xuICAgIH1cbiAgfVxuXG4gIHJldHVybiByYXBpZExpYlRyYWluaW5nU2V0O1xufTtcblxuLypcbiAqIEBtb2R1bGUgeG1tXG4gKlxuICogQGRlc2NyaXB0aW9uIEFsbCB0aGUgZm9sbG93aW5nIGZ1bmN0aW9ucyBjb252ZXJ0IGZyb20gLyB0byByYXBpZE1peCAvIFhNTSBKU09OIG9iamVjdHMuXG4gKi9cblxuLyoqXG4gKiBDb252ZXJ0IGEgUkFQSUQtTUlYIHRyYWluaW5nIHNldCBPYmplY3QgdG8gYW4gWE1NIHRyYWluaW5nIHNldCBPYmplY3QuXG4gKlxuICogQHBhcmFtIHtKU09OfSByYXBpZE1peFRyYWluaW5nU2V0IC0gQSBSQVBJRC1NSVggY29tcGF0aWJsZSB0cmFpbmluZyBzZXRcbiAqXG4gKiBAcmV0dXJuIHtKU09OfSB4bW1UcmFpbmluZ1NldCAtIEFuIFhNTSBjb21wYXRpYmxlIHRyYWluaW5nIHNldFxuICovXG5jb25zdCByYXBpZE1peFRvWG1tVHJhaW5pbmdTZXQgPSByYXBpZE1peFRyYWluaW5nU2V0ID0+IHtcbiAgY29uc3QgcGF5bG9hZCA9IHJhcGlkTWl4VHJhaW5pbmdTZXQucGF5bG9hZDtcblxuICBjb25zdCBjb25maWcgPSB7XG4gICAgYmltb2RhbDogcGF5bG9hZC5vdXRwdXREaW1lbnNpb24gPiAwLFxuICAgIGRpbWVuc2lvbjogcGF5bG9hZC5pbnB1dERpbWVuc2lvbiArIHBheWxvYWQub3V0cHV0RGltZW5zaW9uLFxuICAgIGRpbWVuc2lvbklucHV0OiAocGF5bG9hZC5vdXRwdXREaW1lbnNpb24gPiAwKSA/IHBheWxvYWQuaW5wdXREaW1lbnNpb24gOiAwLFxuICB9O1xuXG4gIGlmIChwYXlsb2FkLmNvbHVtbk5hbWVzKSB7XG4gICAgY29uZmlnLmNvbHVtbk5hbWVzID0gcGF5bG9hZC5jb2x1bW5OYW1lcy5pbnB1dC5zbGljZSgpO1xuICAgIGNvbmZpZy5jb2x1bW5OYW1lcyA9IGNvbmZpZy5jb2x1bW5OYW1lcy5jb25jYXQocGF5bG9hZC5jb2x1bW5OYW1lcy5vdXRwdXQpO1xuICB9XG5cbiAgY29uc3QgcGhyYXNlTWFrZXIgPSBuZXcgWG1tLlBocmFzZU1ha2VyKGNvbmZpZyk7XG4gIGNvbnN0IHNldE1ha2VyID0gbmV3IFhtbS5TZXRNYWtlcigpO1xuXG4gIGZvciAobGV0IGkgPSAwOyBpIDwgcGF5bG9hZC5kYXRhLmxlbmd0aDsgaSsrKSB7XG4gICAgY29uc3QgZGF0dW0gPSBwYXlsb2FkLmRhdGFbaV07XG5cbiAgICBwaHJhc2VNYWtlci5yZXNldCgpO1xuICAgIHBocmFzZU1ha2VyLnNldENvbmZpZyh7IGxhYmVsOiBkYXR1bS5sYWJlbCB9KTtcblxuICAgIGZvciAobGV0IGogPSAwOyBqIDwgZGF0dW0uaW5wdXQubGVuZ3RoOyBqKyspIHtcbiAgICAgIGxldCB2ZWN0b3IgPSBkYXR1bS5pbnB1dFtqXTtcblxuICAgICAgaWYgKHBheWxvYWQub3V0cHV0RGltZW5zaW9uID4gMClcbiAgICAgICAgdmVjdG9yID0gdmVjdG9yLmNvbmNhdChkYXR1bS5vdXRwdXRbal0pO1xuXG4gICAgICBwaHJhc2VNYWtlci5hZGRPYnNlcnZhdGlvbih2ZWN0b3IpO1xuICAgIH1cblxuICAgIHNldE1ha2VyLmFkZFBocmFzZShwaHJhc2VNYWtlci5nZXRQaHJhc2UoKSk7XG4gIH1cblxuICByZXR1cm4gc2V0TWFrZXIuZ2V0VHJhaW5pbmdTZXQoKTtcbn1cblxuLyoqXG4gKiBDb252ZXJ0IGFuIFhNTSB0cmFpbmluZyBzZXQgT2JqZWN0IHRvIGEgUkFQSUQtTUlYIHRyYWluaW5nIHNldCBPYmplY3QuXG4gKlxuICogQHBhcmFtIHtKU09OfSB4bW1UcmFpbmluZ1NldCAtIEFuIFhNTSBjb21wYXRpYmxlIHRyYWluaW5nIHNldFxuICpcbiAqIEByZXR1cm4ge0pTT059IHJhcGlkTWl4VHJhaW5pbmdTZXQgLSBBIFJBUElELU1JWCBjb21wYXRpYmxlIHRyYWluaW5nIHNldFxuICovXG5jb25zdCB4bW1Ub1JhcGlkTWl4VHJhaW5pbmdTZXQgPSB4bW1UcmFpbmluZ1NldCA9PiB7XG4gIGNvbnN0IHBheWxvYWQgPSB7XG4gICAgY29sdW1uTmFtZXM6IHsgaW5wdXQ6IFtdLCBvdXRwdXQ6IFtdIH0sXG4gICAgZGF0YTogW11cbiAgfTtcbiAgY29uc3QgcGhyYXNlcyA9IHhtbVRyYWluaW5nU2V0LnBocmFzZXM7XG5cbiAgaWYgKHhtbVRyYWluaW5nU2V0LmJpbW9kYWwpIHtcbiAgICBwYXlsb2FkLmlucHV0RGltZW5zaW9uID0geG1tVHJhaW5pbmdTZXQuZGltZW5zaW9uX2lucHV0O1xuICAgIHBheWxvYWQub3V0cHV0RGltZW5zaW9uID0geG1tVHJhaW5pbmdTZXQuZGltZW5zaW9uIC0geG1tVHJhaW5pbmdTZXQuZGltZW5zaW9uX2lucHV0O1xuXG4gICAgY29uc3QgaURpbSA9IHBheWxvYWQuaW5wdXREaW1lbnNpb247XG4gICAgY29uc3Qgb0RpbSA9IHBheWxvYWQub3V0cHV0RGltZW5zaW9uO1xuXG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB4bW1UcmFpbmluZ1NldC5jb2x1bW5fbmFtZXMubGVuZ3RoOyBpKyspIHtcbiAgICAgIGlmIChpIDwgaURpbSkge1xuICAgICAgICBwYXlsb2FkLmNvbHVtbk5hbWVzLmlucHV0LnB1c2goeG1tVHJhaW5pbmdTZXQuY29sdW1uX25hbWVzW2ldKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHBheWxvYWQuY29sdW1uTmFtZXMub3V0cHV0LnB1c2goeG1tVHJhaW5pbmdTZXQuY29sdW1uX25hbWVzW2ldKTtcbiAgICAgIH1cbiAgICB9XG5cbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHBocmFzZXMubGVuZ3RoOyBpKyspIHtcbiAgICAgIGNvbnN0IGV4YW1wbGUgPSB7XG4gICAgICAgIGlucHV0OltdLFxuICAgICAgICBvdXRwdXQ6IFtdLFxuICAgICAgICBsYWJlbDogcGhyYXNlc1tpXS5sYWJlbFxuICAgICAgfTtcblxuICAgICAgZm9yIChsZXQgaiA9IDA7IGogPCBwaHJhc2VzW2ldLmxlbmd0aDsgaisrKSB7XG4gICAgICAgIGV4YW1wbGUuaW5wdXQucHVzaChwaHJhc2VzW2ldLmRhdGFfaW5wdXQuc2xpY2UoaiAqIGlEaW0sIChqICsgMSkgKiBpRGltKSk7XG4gICAgICAgIGV4YW1wbGUub3V0cHV0LnB1c2gocGhyYXNlc1tpXS5kYXRhX291dHB1dC5zbGljZShqICogb0RpbSwgKGogKyAxKSAqIG9EaW0pKTtcbiAgICAgIH1cblxuICAgICAgcGF5bG9hZC5kYXRhLnB1c2goZXhhbXBsZSk7XG4gICAgfVxuICB9IGVsc2Uge1xuICAgIHBheWxvYWQuaW5wdXREaW1lbnNpb24gPSB4bW1UcmFpbmluZ1NldC5kaW1lbnNpb247XG4gICAgcGF5bG9hZC5vdXRwdXREaW1lbnNpb24gPSAwO1xuXG4gICAgY29uc3QgZGltID0gcGF5bG9hZC5pbnB1dERpbWVuc2lvbjtcblxuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgeG1tVHJhaW5pbmdTZXQuY29sdW1uX25hbWVzLmxlbmd0aDsgaSsrKSB7XG4gICAgICBwYXlsb2FkLmNvbHVtbk5hbWVzLmlucHV0LnB1c2goeG1tVHJhaW5pbmdTZXQuY29sdW1uX25hbWVzW2ldKTtcbiAgICB9XG5cbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHBocmFzZXMubGVuZ3RoOyBpKyspIHtcbiAgICAgIGNvbnN0IGV4YW1wbGUgPSB7XG4gICAgICAgIGlucHV0OltdLFxuICAgICAgICBvdXRwdXQ6IFtdLFxuICAgICAgICBsYWJlbDogcGhyYXNlc1tpXS5sYWJlbFxuICAgICAgfTtcblxuICAgICAgZm9yIChsZXQgaiA9IDA7IGogPCBwaHJhc2VzW2ldLmxlbmd0aDsgaisrKSB7XG4gICAgICAgIGV4YW1wbGUuaW5wdXQucHVzaChwaHJhc2VzW2ldLmRhdGEuc2xpY2UoaiAqIGRpbSwgKGogKyAxKSAqIGRpbSkpO1xuICAgICAgfVxuXG4gICAgICBwYXlsb2FkLmRhdGEucHVzaChleGFtcGxlKTtcbiAgICB9XG4gIH1cblxuICByZXR1cm4ge1xuICAgIGRvY1R5cGU6ICdyYXBpZC1taXg6bWwtdHJhaW5pbmctc2V0JyxcbiAgICBkb2NWZXJzaW9uOiBSQVBJRF9NSVhfRE9DX1ZFUlNJT04sXG4gICAgcGF5bG9hZDogcGF5bG9hZCxcbiAgfTtcbn1cblxuLyoqXG4gKiBDb252ZXJ0IGEgUkFQSUQtTUlYIGNvbmZpZ3VyYXRpb24gT2JqZWN0IHRvIGFuIFhNTSBjb25maWd1cmF0aW9uIE9iamVjdC5cbiAqXG4gKiBAcGFyYW0ge0pTT059IHJhcGlkTWl4Q29uZmlnIC0gQSBSQVBJRC1NSVggY29tcGF0aWJsZSBjb25maWd1cmFpdG9uIG9iamVjdFxuICpcbiAqIEByZXR1cm4ge0pTT059IHhtbUNvbmZpZyAtIEEgY29uZmlndXJhdGlvbiBvYmplY3QgcmVhZHkgdG8gYmUgdXNlZCBieSB0aGUgWE1NIGxpYnJhcnlcbiAqL1xuY29uc3QgcmFwaWRNaXhUb1htbUNvbmZpZyA9IHJhcGlkTWl4Q29uZmlnID0+IHtcbiAgcmV0dXJuIHJhcGlkTWl4Q29uZmlnLnBheWxvYWQ7XG59XG5cbi8qKlxuICogQ29udmVydCBhbiBYTU0gY29uZmlndXJhdGlvbiBPYmplY3QgdG8gYSBSQVBJRC1NSVggY29uZmlndXJhdGlvbiBzZXQgT2JqZWN0LlxuICpcbiAqIEBwYXJhbSB7SlNPTn0geG1tQ29uZmlnIC0gQSBjb25maWd1cmF0aW9uIG9iamVjdCB0YXJnZXRpbmcgdGhlIFhNTSBsaWJyYXJ5XG4gKlxuICogQHJldHVybiB7SlNPTn0gcmFwaWRNaXhDb25maWcgLSBBIFJBUElELU1JWCBjb21wYXRpYmxlIGNvbmZpZ3VyYXRpb24gb2JqZWN0XG4gKi9cbmNvbnN0IHhtbVRvUmFwaWRNaXhDb25maWcgPSB4bW1Db25maWcgPT4ge1xuICByZXR1cm4ge1xuICAgIGRvY1R5cGU6ICdyYXBpZC1taXg6bWwtY29uZmlndXJhdGlvbicsXG4gICAgZG9jVmVyc2lvbjogUkFQSURfTUlYX0RPQ19WRVJTSU9OLFxuICAgIHRhcmdldDoge1xuICAgICAgbmFtZTogYHhtbToke3htbUNvbmZpZy5tb2RlbFR5cGV9YCxcbiAgICAgIHZlcnNpb246ICcxLjAuMCcsXG4gICAgfSxcbiAgICBwYXlsb2FkOiB4bW1Db25maWcsXG4gIH1cbn1cblxuLyoqXG4gKiBDb252ZXJ0IGEgUkFQSUQtTUlYIGNvbmZpZ3VyYXRpb24gT2JqZWN0IHRvIGFuIFhNTSBjb25maWd1cmF0aW9uIE9iamVjdC5cbiAqXG4gKiBAcGFyYW0ge0pTT059IHJhcGlkTWl4TW9kZWwgLSBBIFJBUElELU1JWCBjb21wYXRpYmxlIG1vZGVsXG4gKlxuICogQHJldHVybiB7SlNPTn0geG1tTW9kZWwgLSBBIG1vZGVsIHJlYWR5IHRvIGJlIHVzZWQgYnkgdGhlIFhNTSBsaWJyYXJ5XG4gKi9cbmNvbnN0IHJhcGlkTWl4VG9YbW1Nb2RlbCA9IHJhcGlkTWl4TW9kZWwgPT4ge1xuICByZXR1cm4gcmFwaWRNaXhNb2RlbC5wYXlsb2FkO1xufVxuXG4vKipcbiAqIENvbnZlcnQgYW4gWE1NIG1vZGVsIE9iamVjdCB0byBhIFJBUElELU1JWCBtb2RlbCBPYmplY3QuXG4gKlxuICogQHBhcmFtIHtKU09OfSB4bW1Nb2RlbCAtIEEgbW9kZWwgZ2VuZXJhdGVkIGJ5IHRoZSBYTU0gbGlicmFyeVxuICpcbiAqIEByZXR1cm4ge0pTT059IHJhcGlkTWl4TW9kZWwgLSBBIFJBUElELU1JWCBjb21wYXRpYmxlIG1vZGVsXG4gKi9cbmNvbnN0IHhtbVRvUmFwaWRNaXhNb2RlbCA9IHhtbU1vZGVsID0+IHtcbiAgY29uc3QgbW9kZWxUeXBlID0geG1tTW9kZWwuY29uZmlndXJhdGlvbi5kZWZhdWx0X3BhcmFtZXRlcnMuc3RhdGVzID8gJ2hobW0nIDogJ2dtbSc7XG5cbiAgcmV0dXJuIHtcbiAgICBkb2NUeXBlOiAncmFwaWQtbWl4Om1sLW1vZGVsJyxcbiAgICBkb2NWZXJzaW9uOiBSQVBJRF9NSVhfRE9DX1ZFUlNJT04sXG4gICAgdGFyZ2V0OiB7XG4gICAgICBuYW1lOiBgeG1tOiR7bW9kZWxUeXBlfWAsXG4gICAgICB2ZXJzaW9uOiAnMS4wLjAnLFxuICAgIH0sXG4gICAgcGF5bG9hZDogeG1tTW9kZWwsXG4gIH1cbn07XG5cbi8qXG4gKiBAbW9kdWxlIGNvbW9cbiAqXG4gKiBAZGVzY3JpcHRpb24gRm9yIHRoZSBtb21lbnQgdGhlIGNvbW8gd2ViIHNlcnZpY2Ugd2lsbCBvbmx5IHJldHVybiBYTU0gbW9kZWxzXG4gKiB3cmFwcGVkIGludG8gUkFQSUQtTUlYIEpTT04gb2JqZWN0cywgdGFraW5nIFJBUElELU1JWCB0cmFpbmluZ3Mgc2V0cyBhbmQgWE1NIGNvbmZpZ3VyYXRpb25zLlxuICovXG5cbi8qKlxuICogQ3JlYXRlIHRoZSBKU09OIHRvIHNlbmQgdG8gdGhlIENvbW8gd2ViIHNlcnZpY2UgdmlhIGh0dHAgcmVxdWVzdC5cbiAqXG4gKiBAcGFyYW0ge0pTT059IGNvbmZpZyAtIEEgdmFsaWQgUkFQSUQtTUlYIGNvbmZpZ3VyYXRpb24gb2JqZWN0XG4gKiBAcGFyYW0ge0pTT059IHRyYWluaW5nU2V0IC0gQSB2YWxpZCBSQVBJRC1NSVggdHJhaW5pbmcgc2V0IG9iamVjdFxuICogQHBhcmFtIHtKU09OfSBbbWV0YXM9bnVsbF0gLSBTb21lIG9wdGlvbmFsIG1ldGEgZGF0YVxuICogQHBhcmFtIHtKU09OfSBbc2lnbmFsUHJvY2Vzc2luZz1udWxsXSAtIEFuIG9wdGlvbmFsIGRlc2NyaXB0aW9uIG9mIHRoZSBwcmUgcHJvY2Vzc2luZyB1c2VkIHRvIG9idGFpbiB0aGUgdHJhaW5pbmcgc2V0XG4gKlxuICogQHJldHVybiB7SlNPTn0gaHR0cFJlcXVlc3QgLSBBIHZhbGlkIEpTT04gdG8gYmUgc2VudCB0byB0aGUgQ29tbyB3ZWIgc2VydmljZSB2aWEgaHR0cCByZXF1ZXN0LlxuICovXG5jb25zdCBjcmVhdGVDb21vSHR0cFJlcXVlc3QgPSAoY29uZmlnLCB0cmFpbmluZ1NldCwgbWV0YXMgPSBudWxsLCBzaWduYWxQcm9jZXNzaW5nID0gbnVsbCkgPT4ge1xuICBjb25zdCByZXNxdWVzdCA9IHtcbiAgICBkb2NUeXBlOiAncmFwaWQtbWl4Om1sLWh0dHAtcmVxdWVzdCcsXG4gICAgZG9jVmVyc2lvbjogUkFQSURfTUlYX0RPQ19WRVJTSU9OLFxuICAgIHRhcmdldDoge1xuICAgICAgbmFtZTogJ2NvbW8td2ViLXNlcnZpY2UnLFxuICAgICAgdmVyc2lvbjogJzEuMC4wJ1xuICAgIH0sXG4gICAgcGF5bG9hZDoge1xuICAgICAgY29uZmlndXJhdGlvbjogY29uZmlnLFxuICAgICAgdHJhaW5pbmdTZXQ6IHRyYWluaW5nU2V0XG4gICAgfVxuICB9O1xuXG4gIGlmIChtZXRhcyAhPT0gbnVsbCkge1xuICAgIHJlc3F1ZXN0LnBheWxvYWQubWV0YXMgPSBtZXRhcztcbiAgfVxuXG4gIGlmIChzaWduYWxQcm9jZXNzaW5nICE9PSBudWxsKSB7XG4gICAgcmVzcXVlc3QucGF5bG9hZC5zaWduYWxQcm9jZXNzaW5nID0gc2lnbmFsUHJvY2Vzc2luZztcbiAgfVxuXG4gIHJldHVybiByZXNxdWVzdDtcbn07XG5cbi8qKlxuICogQ3JlYXRlIHRoZSBKU09OIHRvIHNlbmQgYmFjayBhcyBhIHJlc3BvbnNlIHRvIGh0dHAgcmVxdWVzdHMgdG8gdGhlIENvbW8gd2ViIHNlcnZpY2UuXG4gKlxuICogQHBhcmFtIHtKU09OfSBjb25maWcgLSBBIHZhbGlkIFJBUElELU1JWCBjb25maWd1cmF0aW9uIG9iamVjdFxuICogQHBhcmFtIHtKU09OfSBtb2RlbCAtIEEgdmFsaWQgUkFQSUQtTUlYIG1vZGVsIG9iamVjdFxuICogQHBhcmFtIHtKU09OfSBbbWV0YXM9bnVsbF0gLSBTb21lIG9wdGlvbmFsIG1ldGEgZGF0YVxuICogQHBhcmFtIHtKU09OfSBbc2lnbmFsUHJvY2Vzc2luZz1udWxsXSAtIEFuIG9wdGlvbmFsIGRlc2NyaXB0aW9uIG9mIHRoZSBwcmUgcHJvY2Vzc2luZyB1c2VkIHRvIG9idGFpbiB0aGUgdHJhaW5pbmcgc2V0XG4gKlxuICogQHJldHVybiB7SlNPTn0gaHR0cFJlc3BvbnNlIC0gQSB2YWxpZCBKU09OIHJlc3BvbnNlIHRvIGJlIHNlbnQgYmFjayBmcm9tIHRoZSBDb21vIHdlYiBzZXJ2aWNlIHZpYSBodHRwLlxuICovXG5jb25zdCBjcmVhdGVDb21vSHR0cFJlc3BvbnNlID0gKGNvbmZpZywgbW9kZWwsIG1ldGFzID0gbnVsbCwgc2lnbmFsUHJvY2Vzc2luZyA9IG51bGwpID0+IHtcbiAgY29uc3QgcmVzcG9uc2UgPSB7XG4gICAgZG9jVHlwZTogJ3JhcGlkLW1peDptbC1odHRwLXJlc3BvbnNlJyxcbiAgICBkb2NWZXJzaW9uOiBSQVBJRF9NSVhfRE9DX1ZFUlNJT04sXG4gICAgdGFyZ2V0OiB7XG4gICAgICBuYW1lOiAnY29tby13ZWItc2VydmljZScsXG4gICAgICB2ZXJzaW9uOiAnMS4wLjAnXG4gICAgfSxcbiAgICBwYXlsb2FkOiB7XG4gICAgICBjb25maWd1cmF0aW9uOiBjb25maWcsXG4gICAgICBtb2RlbDogbW9kZWxcbiAgICB9XG4gIH07XG5cbiAgaWYgKG1ldGFzICE9PSBudWxsKSB7XG4gICAgcmVzcG9uc2UucGF5bG9hZC5tZXRhcyA9IG1ldGFzO1xuICB9XG5cbiAgaWYgKHNpZ25hbFByb2Nlc3NpbmcgIT09IG51bGwpIHtcbiAgICByZXNwb25zZS5wYXlsb2FkLnNpZ25hbFByb2Nlc3NpbmcgPSBzaWduYWxQcm9jZXNzaW5nO1xuICB9XG5cbiAgcmV0dXJuIHJlc3BvbnNlO1xufTtcblxuXG5leHBvcnQgZGVmYXVsdCB7XG4gIC8vIHJhcGlkTGliIGFkYXB0ZXJzXG4gIHJhcGlkTWl4VG9SYXBpZExpYlRyYWluaW5nU2V0LFxuXG4gIC8vIHhtbSBhZGFwdGVyc1xuICByYXBpZE1peFRvWG1tVHJhaW5pbmdTZXQsXG4gIHhtbVRvUmFwaWRNaXhUcmFpbmluZ1NldCxcblxuICByYXBpZE1peFRvWG1tQ29uZmlnLFxuICB4bW1Ub1JhcGlkTWl4Q29uZmlnLFxuXG4gIHJhcGlkTWl4VG9YbW1Nb2RlbCxcbiAgeG1tVG9SYXBpZE1peE1vZGVsLFxuXG4gIGNyZWF0ZUNvbW9IdHRwUmVxdWVzdCxcbiAgY3JlYXRlQ29tb0h0dHBSZXNwb25zZSxcbiAgLy8gY29uc3RhbnRzXG4gIFJBUElEX01JWF9ET0NfVkVSU0lPTixcbiAgUkFQSURfTUlYX0RFRkFVTFRfTEFCRUxcbn07XG4iXX0=