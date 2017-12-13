'use strict';Object.defineProperty(exports,"__esModule",{value:true});var _assign=require('babel-runtime/core-js/object/assign');var _assign2=_interopRequireDefault(_assign);var _xmmClient=require('xmm-client');var Xmm=_interopRequireWildcard(_xmmClient);function _interopRequireWildcard(obj){if(obj&&obj.__esModule){return obj;}else{var newObj={};if(obj!=null){for(var key in obj){if(Object.prototype.hasOwnProperty.call(obj,key))newObj[key]=obj[key];}}newObj.default=obj;return newObj;}}function _interopRequireDefault(obj){return obj&&obj.__esModule?obj:{default:obj};}/*
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
 * @param {Object} rapidMixTrainingSet - A RAPID-MIX compatible training set
 *
 * @return {Object} rapidLibTrainingSet - A RapidLib JS compatible training set
 */var rapidMixToRapidLibTrainingSet=function rapidMixToRapidLibTrainingSet(rapidMixTrainingSet){var rapidLibTrainingSet=[];for(var i=0;i<rapidMixTrainingSet.payload.data.length;i++){var phrase=rapidMixTrainingSet.payload.data[i];for(var j=0;j<phrase.input.length;j++){var el={label:phrase.label,input:phrase.input[j],output:phrase.output.length>0?phrase.output[j]:[]};rapidLibTrainingSet.push(el);}}return rapidLibTrainingSet;};/*
 * @module xmm
 *
 * @description All the following functions convert from / to rapidMix / XMM JSON objects.
 *//**
 * Convert a RAPID-MIX training set Object to an XMM training set Object.
 *
 * @param {Object} rapidMixTrainingSet - A RAPID-MIX compatible training set
 *
 * @return {Object} xmmTrainingSet - An XMM compatible training set
 */var rapidMixToXmmTrainingSet=function rapidMixToXmmTrainingSet(rapidMixTrainingSet){var payload=rapidMixTrainingSet.payload;var config={bimodal:payload.outputDimension>0,dimension:payload.inputDimension+payload.outputDimension,dimensionInput:payload.outputDimension>0?payload.inputDimension:0};if(payload.columnNames){config.columnNames=payload.columnNames.input.slice();config.columnNames=config.columnNames.concat(payload.columnNames.output);}var phraseMaker=new Xmm.PhraseMaker(config);var setMaker=new Xmm.SetMaker();for(var i=0;i<payload.data.length;i++){var datum=payload.data[i];phraseMaker.reset();phraseMaker.setConfig({label:datum.label});for(var j=0;j<datum.input.length;j++){var vector=datum.input[j];if(payload.outputDimension>0)vector=vector.concat(datum.output[j]);phraseMaker.addObservation(vector);}setMaker.addPhrase(phraseMaker.getPhrase());}return setMaker.getTrainingSet();};/**
 * Convert an XMM training set Object to a RAPID-MIX training set Object.
 *
 * @param {Object} xmmTrainingSet - An XMM compatible training set
 *
 * @return {Object} rapidMixTrainingSet - A RAPID-MIX compatible training set
 */var xmmToRapidMixTrainingSet=function xmmToRapidMixTrainingSet(xmmTrainingSet){var payload={columnNames:{input:[],output:[]},data:[]};var phrases=xmmTrainingSet.phrases;if(xmmTrainingSet.bimodal){payload.inputDimension=xmmTrainingSet.dimension_input;payload.outputDimension=xmmTrainingSet.dimension-xmmTrainingSet.dimension_input;var iDim=payload.inputDimension;var oDim=payload.outputDimension;for(var i=0;i<xmmTrainingSet.column_names.length;i++){if(i<iDim){payload.columnNames.input.push(xmmTrainingSet.column_names[i]);}else{payload.columnNames.output.push(xmmTrainingSet.column_names[i]);}}for(var _i=0;_i<phrases.length;_i++){var example={input:[],output:[],label:phrases[_i].label};for(var j=0;j<phrases[_i].length;j++){example.input.push(phrases[_i].data_input.slice(j*iDim,(j+1)*iDim));example.output.push(phrases[_i].data_output.slice(j*oDim,(j+1)*oDim));}payload.data.push(example);}}else{payload.inputDimension=xmmTrainingSet.dimension;payload.outputDimension=0;var dim=payload.inputDimension;for(var _i2=0;_i2<xmmTrainingSet.column_names.length;_i2++){payload.columnNames.input.push(xmmTrainingSet.column_names[_i2]);}for(var _i3=0;_i3<phrases.length;_i3++){var _example={input:[],output:[],label:phrases[_i3].label};for(var _j=0;_j<phrases[_i3].length;_j++){_example.input.push(phrases[_i3].data.slice(_j*dim,(_j+1)*dim));}payload.data.push(_example);}}return{docType:'rapid-mix:ml-training-set',docVersion:RAPID_MIX_DOC_VERSION,payload:payload};};/**
 * Convert a RAPID-MIX configuration Object to an XMM configuration Object.
 *
 * @param {Object} rapidMixConfig - A RAPID-MIX compatible configuraiton object
 *
 * @return {Object} xmmConfig - A configuration object ready to be used by the XMM library
 */var rapidMixToXmmConfig=function rapidMixToXmmConfig(rapidMixConfig){return rapidMixConfig.payload;};/**
 * Convert an XMM configuration Object to a RAPID-MIX configuration set Object.
 *
 * @param {Object} xmmConfig - A configuration object targeting the XMM library
 *
 * @return {Object} rapidMixConfig - A RAPID-MIX compatible configuration object
 */var xmmToRapidMixConfig=function xmmToRapidMixConfig(xmmConfig){return{docType:'rapid-mix:ml-configuration',docVersion:RAPID_MIX_DOC_VERSION,target:{name:'xmm',version:'1.0.0'},payload:xmmConfig};};/**
 * Convert a RAPID-MIX configuration Object to an XMM configuration Object.
 *
 * @param {Object} rapidMixModel - A RAPID-MIX compatible model
 *
 * @return {Object} xmmModel - A model ready to be used by the XMM library
 */var rapidMixToXmmModel=function rapidMixToXmmModel(rapidMixModel){return rapidMixModel.payload;};/**
 * Convert an XMM model Object to a RAPID-MIX model Object.
 *
 * @param {Object} xmmModel - A model generated by the XMM library
 *
 * @return {Object} rapidMixModel - A RAPID-MIX compatible model
 */var xmmToRapidMixModel=function xmmToRapidMixModel(xmmModel){var modelType=xmmModel.configuration.default_parameters.states?'hhmm':'gmm';return{docType:'rapid-mix:ml-model',docVersion:RAPID_MIX_DOC_VERSION,target:{name:'xmm',version:'1.0.0'},payload:(0,_assign2.default)({},xmmModel,{modelType:modelType})};};/*
 * @module como
 *
 * @description For the moment the como web service will only return XMM models
 * wrapped into RAPID-MIX JSON objects, taking RAPID-MIX trainings sets and XMM configurations.
 *//**
 * Create the JSON to send to the Como web service via http request.
 *
 * @param {Object} config - A valid RAPID-MIX configuration object
 * @param {Object} trainingSet - A valid RAPID-MIX training set object
 * @param {Object} [metas=null] - Some optional meta data
 * @param {Object} [signalProcessing=null] - An optional description of the pre processing used to obtain the training set
 *
 * @return {Object} httpRequest - A valid JSON to be sent to the Como web service via http request.
 */var createComoHttpRequest=function createComoHttpRequest(config,trainingSet){var metas=arguments.length>2&&arguments[2]!==undefined?arguments[2]:null;var signalProcessing=arguments.length>3&&arguments[3]!==undefined?arguments[3]:null;var resquest={docType:'rapid-mix:ml-http-request',docVersion:RAPID_MIX_DOC_VERSION,target:{name:'como-web-service',version:'1.0.0'},payload:{configuration:config,trainingSet:trainingSet}};if(metas!==null){resquest.payload.metas=metas;}if(signalProcessing!==null){resquest.payload.signalProcessing=signalProcessing;}return resquest;};/**
 * Create the JSON to send back as a response to http requests to the Como web service.
 *
 * @param {Object} config - A valid RAPID-MIX configuration object
 * @param {Object} model - A valid RAPID-MIX model object
 * @param {Object} [metas=null] - Some optional meta data
 * @param {Object} [signalProcessing=null] - An optional description of the pre processing used to obtain the training set
 *
 * @return {Object} httpResponse - A valid JSON response to be sent back from the Como web service via http.
 */var createComoHttpResponse=function createComoHttpResponse(config,model){var metas=arguments.length>2&&arguments[2]!==undefined?arguments[2]:null;var signalProcessing=arguments.length>3&&arguments[3]!==undefined?arguments[3]:null;var response={docType:'rapid-mix:ml-http-response',docVersion:RAPID_MIX_DOC_VERSION,target:{name:'como-web-service',version:'1.0.0'},payload:{configuration:config,model:model}};if(metas!==null){response.payload.metas=metas;}if(signalProcessing!==null){response.payload.signalProcessing=signalProcessing;}return response;};exports.default={// rapidLib adapters
rapidMixToRapidLibTrainingSet:rapidMixToRapidLibTrainingSet,// xmm adapters
rapidMixToXmmTrainingSet:rapidMixToXmmTrainingSet,xmmToRapidMixTrainingSet:xmmToRapidMixTrainingSet,rapidMixToXmmConfig:rapidMixToXmmConfig,xmmToRapidMixConfig:xmmToRapidMixConfig,rapidMixToXmmModel:rapidMixToXmmModel,xmmToRapidMixModel:xmmToRapidMixModel,createComoHttpRequest:createComoHttpRequest,createComoHttpResponse:createComoHttpResponse,// constants
RAPID_MIX_DOC_VERSION:RAPID_MIX_DOC_VERSION,RAPID_MIX_DEFAULT_LABEL:RAPID_MIX_DEFAULT_LABEL};
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImluZGV4LmpzIl0sIm5hbWVzIjpbIlhtbSIsIlJBUElEX01JWF9ET0NfVkVSU0lPTiIsIlJBUElEX01JWF9ERUZBVUxUX0xBQkVMIiwicmFwaWRNaXhUb1JhcGlkTGliVHJhaW5pbmdTZXQiLCJyYXBpZExpYlRyYWluaW5nU2V0IiwiaSIsInJhcGlkTWl4VHJhaW5pbmdTZXQiLCJwYXlsb2FkIiwiZGF0YSIsImxlbmd0aCIsInBocmFzZSIsImoiLCJpbnB1dCIsImVsIiwibGFiZWwiLCJvdXRwdXQiLCJwdXNoIiwicmFwaWRNaXhUb1htbVRyYWluaW5nU2V0IiwiY29uZmlnIiwiYmltb2RhbCIsIm91dHB1dERpbWVuc2lvbiIsImRpbWVuc2lvbiIsImlucHV0RGltZW5zaW9uIiwiZGltZW5zaW9uSW5wdXQiLCJjb2x1bW5OYW1lcyIsInNsaWNlIiwiY29uY2F0IiwicGhyYXNlTWFrZXIiLCJQaHJhc2VNYWtlciIsInNldE1ha2VyIiwiU2V0TWFrZXIiLCJkYXR1bSIsInJlc2V0Iiwic2V0Q29uZmlnIiwidmVjdG9yIiwiYWRkT2JzZXJ2YXRpb24iLCJhZGRQaHJhc2UiLCJnZXRQaHJhc2UiLCJnZXRUcmFpbmluZ1NldCIsInhtbVRvUmFwaWRNaXhUcmFpbmluZ1NldCIsInBocmFzZXMiLCJ4bW1UcmFpbmluZ1NldCIsImRpbWVuc2lvbl9pbnB1dCIsImlEaW0iLCJvRGltIiwiY29sdW1uX25hbWVzIiwiZXhhbXBsZSIsImRhdGFfaW5wdXQiLCJkYXRhX291dHB1dCIsImRpbSIsImRvY1R5cGUiLCJkb2NWZXJzaW9uIiwicmFwaWRNaXhUb1htbUNvbmZpZyIsInJhcGlkTWl4Q29uZmlnIiwieG1tVG9SYXBpZE1peENvbmZpZyIsInRhcmdldCIsIm5hbWUiLCJ2ZXJzaW9uIiwieG1tQ29uZmlnIiwicmFwaWRNaXhUb1htbU1vZGVsIiwicmFwaWRNaXhNb2RlbCIsInhtbVRvUmFwaWRNaXhNb2RlbCIsIm1vZGVsVHlwZSIsInhtbU1vZGVsIiwiY29uZmlndXJhdGlvbiIsImRlZmF1bHRfcGFyYW1ldGVycyIsInN0YXRlcyIsImNyZWF0ZUNvbW9IdHRwUmVxdWVzdCIsInRyYWluaW5nU2V0IiwibWV0YXMiLCJzaWduYWxQcm9jZXNzaW5nIiwicmVzcXVlc3QiLCJjcmVhdGVDb21vSHR0cFJlc3BvbnNlIiwibW9kZWwiLCJyZXNwb25zZSJdLCJtYXBwaW5ncyI6IjhLQUFBLHFDLEdBQVlBLEksa1dBRVo7Ozs7R0FNQTs7Ozs7R0FNQSxHQUFNQyx1QkFBd0IsT0FBOUIsQ0FFQTs7Ozs7R0FNQSxHQUFNQyx5QkFBMEIsc0JBQWhDLENBRUE7Ozs7R0FNQTs7Ozs7O0dBT0EsR0FBTUMsK0JBQWdDLFFBQWhDQSw4QkFBZ0MscUJBQXVCLENBQzNELEdBQU1DLHFCQUFzQixFQUE1QixDQUVBLElBQUssR0FBSUMsR0FBSSxDQUFiLENBQWdCQSxFQUFJQyxvQkFBb0JDLE9BQXBCLENBQTRCQyxJQUE1QixDQUFpQ0MsTUFBckQsQ0FBNkRKLEdBQTdELENBQWtFLENBQ2hFLEdBQU1LLFFBQVNKLG9CQUFvQkMsT0FBcEIsQ0FBNEJDLElBQTVCLENBQWlDSCxDQUFqQyxDQUFmLENBRUEsSUFBSyxHQUFJTSxHQUFJLENBQWIsQ0FBZ0JBLEVBQUlELE9BQU9FLEtBQVAsQ0FBYUgsTUFBakMsQ0FBeUNFLEdBQXpDLENBQThDLENBQzVDLEdBQU1FLElBQUssQ0FDVEMsTUFBT0osT0FBT0ksS0FETCxDQUVURixNQUFPRixPQUFPRSxLQUFQLENBQWFELENBQWIsQ0FGRSxDQUdUSSxPQUFRTCxPQUFPSyxNQUFQLENBQWNOLE1BQWQsQ0FBdUIsQ0FBdkIsQ0FBMkJDLE9BQU9LLE1BQVAsQ0FBY0osQ0FBZCxDQUEzQixDQUE4QyxFQUg3QyxDQUFYLENBTUFQLG9CQUFvQlksSUFBcEIsQ0FBeUJILEVBQXpCLEVBQ0QsQ0FDRixDQUVELE1BQU9ULG9CQUFQLENBQ0QsQ0FsQkQsQ0FvQkE7Ozs7R0FNQTs7Ozs7O0dBT0EsR0FBTWEsMEJBQTJCLFFBQTNCQSx5QkFBMkIscUJBQXVCLENBQ3RELEdB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QSxHQUFJSixRQUFVLElBQWQsQ0FBb0IsQ0FDbEJLLFNBQVNuRSxPQUFULENBQWlCOEQsS0FBakIsQ0FBeUJBLEtBQXpCLENBQ0QsQ0FFRCxHQUFJQyxtQkFBcUIsSUFBekIsQ0FBK0IsQ0FDN0JJLFNBQVNuRSxPQUFULENBQWlCK0QsZ0JBQWpCLENBQW9DQSxnQkFBcEMsQ0FDRCxDQUVELE1BQU9JLFNBQVAsQ0FDRCxDQXZCRCxDLGdCQTBCZSxDQUNiO0FBQ0F2RSwyREFGYSxDQUliO0FBQ0FjLGlEQUxhLENBTWJzQixpREFOYSxDQVFiYSx1Q0FSYSxDQVNiRSx1Q0FUYSxDQVdiSyxxQ0FYYSxDQVliRSxxQ0FaYSxDQWNiTSwyQ0FkYSxDQWViSyw2Q0FmYSxDQWdCYjtBQUNBdkUsMkNBakJhLENBa0JiQywrQ0FsQmEsQyIsImZpbGUiOiJpbmRleC5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCAqIGFzIFhtbSBmcm9tICd4bW0tY2xpZW50JztcblxuLypcbiAqIEBtb2R1bGUgY29uc3RhbnRzXG4gKlxuICogQGRlc2NyaXB0aW9uIENvbnN0YW50cyB1c2VkIGJ5IHRoZSBSQVBJRC1NSVggSlNPTiBzcGVjaWZpY2F0aW9uLlxuICovXG5cbi8qKlxuICogQGNvbnN0YW50XG4gKiBAdHlwZSB7U3RyaW5nfVxuICogQGRlc2NyaXB0aW9uIFRoZSBSQVBJRC1NSVggSlNPTiBkb2N1bWVudCBzcGVjaWZpY2F0aW9uIHZlcnNpb24uXG4gKiBAZGVmYXVsdFxuICovXG5jb25zdCBSQVBJRF9NSVhfRE9DX1ZFUlNJT04gPSAnMS4wLjAnO1xuXG4vKipcbiAqIEBjb25zdGFudFxuICogQHR5cGUge1N0cmluZ31cbiAqIEBkZXNjcmlwdGlvbiBUaGUgZGVmYXVsdCBSQVBJRC1NSVggbGFiZWwgdXNlZCB0byBidWlsZCB0cmFpbmluZyBzZXRzLlxuICogQGRlZmF1bHRcbiAqL1xuY29uc3QgUkFQSURfTUlYX0RFRkFVTFRfTEFCRUwgPSAncmFwaWRtaXhEZWZhdWx0TGFiZWwnO1xuXG4vKlxuICogQG1vZHVsZSByYXBpZGxpYlxuICpcbiAqIEBkZXNjcmlwdGlvbiBBbGwgdGhlIGZvbGxvd2luZyBmdW5jdGlvbnMgY29udmVydCBmcm9tIC8gdG8gUkFQSUQtTUlYIC8gUmFwaWRMaWIgSlMgSlNPTiBvYmplY3RzLlxuICovXG5cbi8qKlxuICogQ29udmVydCBhIFJBUElELU1JWCB0cmFpbmluZyBzZXQgT2JqZWN0IHRvIGEgUmFwaWRMaWIgSlMgdHJhaW5pbmcgc2V0IE9iamVjdC5cbiAqXG4gKiBAcGFyYW0ge09iamVjdH0gcmFwaWRNaXhUcmFpbmluZ1NldCAtIEEgUkFQSUQtTUlYIGNvbXBhdGlibGUgdHJhaW5pbmcgc2V0XG4gKlxuICogQHJldHVybiB7T2JqZWN0fSByYXBpZExpYlRyYWluaW5nU2V0IC0gQSBSYXBpZExpYiBKUyBjb21wYXRpYmxlIHRyYWluaW5nIHNldFxuICovXG5jb25zdCByYXBpZE1peFRvUmFwaWRMaWJUcmFpbmluZ1NldCA9IHJhcGlkTWl4VHJhaW5pbmdTZXQgPT4ge1xuICBjb25zdCByYXBpZExpYlRyYWluaW5nU2V0ID0gW107XG5cbiAgZm9yIChsZXQgaSA9IDA7IGkgPCByYXBpZE1peFRyYWluaW5nU2V0LnBheWxvYWQuZGF0YS5sZW5ndGg7IGkrKykge1xuICAgIGNvbnN0IHBocmFzZSA9IHJhcGlkTWl4VHJhaW5pbmdTZXQucGF5bG9hZC5kYXRhW2ldO1xuXG4gICAgZm9yIChsZXQgaiA9IDA7IGogPCBwaHJhc2UuaW5wdXQubGVuZ3RoOyBqKyspIHtcbiAgICAgIGNvbnN0IGVsID0ge1xuICAgICAgICBsYWJlbDogcGhyYXNlLmxhYmVsLFxuICAgICAgICBpbnB1dDogcGhyYXNlLmlucHV0W2pdLFxuICAgICAgICBvdXRwdXQ6IHBocmFzZS5vdXRwdXQubGVuZ3RoID4gMCA/IHBocmFzZS5vdXRwdXRbal0gOiBbXSxcbiAgICAgIH07XG5cbiAgICAgIHJhcGlkTGliVHJhaW5pbmdTZXQucHVzaChlbCk7XG4gICAgfVxuICB9XG5cbiAgcmV0dXJuIHJhcGlkTGliVHJhaW5pbmdTZXQ7XG59O1xuXG4vKlxuICogQG1vZHVsZSB4bW1cbiAqXG4gKiBAZGVzY3JpcHRpb24gQWxsIHRoZSBmb2xsb3dpbmcgZnVuY3Rpb25zIGNvbnZlcnQgZnJvbSAvIHRvIHJhcGlkTWl4IC8gWE1NIEpTT04gb2JqZWN0cy5cbiAqL1xuXG4vKipcbiAqIENvbnZlcnQgYSBSQVBJRC1NSVggdHJhaW5pbmcgc2V0IE9iamVjdCB0byBhbiBYTU0gdHJhaW5pbmcgc2V0IE9iamVjdC5cbiAqXG4gKiBAcGFyYW0ge09iamVjdH0gcmFwaWRNaXhUcmFpbmluZ1NldCAtIEEgUkFQSUQtTUlYIGNvbXBhdGlibGUgdHJhaW5pbmcgc2V0XG4gKlxuICogQHJldHVybiB7T2JqZWN0fSB4bW1UcmFpbmluZ1NldCAtIEFuIFhNTSBjb21wYXRpYmxlIHRyYWluaW5nIHNldFxuICovXG5jb25zdCByYXBpZE1peFRvWG1tVHJhaW5pbmdTZXQgPSByYXBpZE1peFRyYWluaW5nU2V0ID0+IHtcbiAgY29uc3QgcGF5bG9hZCA9IHJhcGlkTWl4VHJhaW5pbmdTZXQucGF5bG9hZDtcblxuICBjb25zdCBjb25maWcgPSB7XG4gICAgYmltb2RhbDogcGF5bG9hZC5vdXRwdXREaW1lbnNpb24gPiAwLFxuICAgIGRpbWVuc2lvbjogcGF5bG9hZC5pbnB1dERpbWVuc2lvbiArIHBheWxvYWQub3V0cHV0RGltZW5zaW9uLFxuICAgIGRpbWVuc2lvbklucHV0OiAocGF5bG9hZC5vdXRwdXREaW1lbnNpb24gPiAwKSA/IHBheWxvYWQuaW5wdXREaW1lbnNpb24gOiAwLFxuICB9O1xuXG4gIGlmIChwYXlsb2FkLmNvbHVtbk5hbWVzKSB7XG4gICAgY29uZmlnLmNvbHVtbk5hbWVzID0gcGF5bG9hZC5jb2x1bW5OYW1lcy5pbnB1dC5zbGljZSgpO1xuICAgIGNvbmZpZy5jb2x1bW5OYW1lcyA9IGNvbmZpZy5jb2x1bW5OYW1lcy5jb25jYXQocGF5bG9hZC5jb2x1bW5OYW1lcy5vdXRwdXQpO1xuICB9XG5cbiAgY29uc3QgcGhyYXNlTWFrZXIgPSBuZXcgWG1tLlBocmFzZU1ha2VyKGNvbmZpZyk7XG4gIGNvbnN0IHNldE1ha2VyID0gbmV3IFhtbS5TZXRNYWtlcigpO1xuXG4gIGZvciAobGV0IGkgPSAwOyBpIDwgcGF5bG9hZC5kYXRhLmxlbmd0aDsgaSsrKSB7XG4gICAgY29uc3QgZGF0dW0gPSBwYXlsb2FkLmRhdGFbaV07XG5cbiAgICBwaHJhc2VNYWtlci5yZXNldCgpO1xuICAgIHBocmFzZU1ha2VyLnNldENvbmZpZyh7IGxhYmVsOiBkYXR1bS5sYWJlbCB9KTtcblxuICAgIGZvciAobGV0IGogPSAwOyBqIDwgZGF0dW0uaW5wdXQubGVuZ3RoOyBqKyspIHtcbiAgICAgIGxldCB2ZWN0b3IgPSBkYXR1bS5pbnB1dFtqXTtcblxuICAgICAgaWYgKHBheWxvYWQub3V0cHV0RGltZW5zaW9uID4gMClcbiAgICAgICAgdmVjdG9yID0gdmVjdG9yLmNvbmNhdChkYXR1bS5vdXRwdXRbal0pO1xuXG4gICAgICBwaHJhc2VNYWtlci5hZGRPYnNlcnZhdGlvbih2ZWN0b3IpO1xuICAgIH1cblxuICAgIHNldE1ha2VyLmFkZFBocmFzZShwaHJhc2VNYWtlci5nZXRQaHJhc2UoKSk7XG4gIH1cblxuICByZXR1cm4gc2V0TWFrZXIuZ2V0VHJhaW5pbmdTZXQoKTtcbn1cblxuLyoqXG4gKiBDb252ZXJ0IGFuIFhNTSB0cmFpbmluZyBzZXQgT2JqZWN0IHRvIGEgUkFQSUQtTUlYIHRyYWluaW5nIHNldCBPYmplY3QuXG4gKlxuICogQHBhcmFtIHtPYmplY3R9IHhtbVRyYWluaW5nU2V0IC0gQW4gWE1NIGNvbXBhdGlibGUgdHJhaW5pbmcgc2V0XG4gKlxuICogQHJldHVybiB7T2JqZWN0fSByYXBpZE1peFRyYWluaW5nU2V0IC0gQSBSQVBJRC1NSVggY29tcGF0aWJsZSB0cmFpbmluZyBzZXRcbiAqL1xuY29uc3QgeG1tVG9SYXBpZE1peFRyYWluaW5nU2V0ID0geG1tVHJhaW5pbmdTZXQgPT4ge1xuICBjb25zdCBwYXlsb2FkID0ge1xuICAgIGNvbHVtbk5hbWVzOiB7IGlucHV0OiBbXSwgb3V0cHV0OiBbXSB9LFxuICAgIGRhdGE6IFtdXG4gIH07XG4gIGNvbnN0IHBocmFzZXMgPSB4bW1UcmFpbmluZ1NldC5waHJhc2VzO1xuXG4gIGlmICh4bW1UcmFpbmluZ1NldC5iaW1vZGFsKSB7XG4gICAgcGF5bG9hZC5pbnB1dERpbWVuc2lvbiA9IHhtbVRyYWluaW5nU2V0LmRpbWVuc2lvbl9pbnB1dDtcbiAgICBwYXlsb2FkLm91dHB1dERpbWVuc2lvbiA9IHhtbVRyYWluaW5nU2V0LmRpbWVuc2lvbiAtIHhtbVRyYWluaW5nU2V0LmRpbWVuc2lvbl9pbnB1dDtcblxuICAgIGNvbnN0IGlEaW0gPSBwYXlsb2FkLmlucHV0RGltZW5zaW9uO1xuICAgIGNvbnN0IG9EaW0gPSBwYXlsb2FkLm91dHB1dERpbWVuc2lvbjtcblxuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgeG1tVHJhaW5pbmdTZXQuY29sdW1uX25hbWVzLmxlbmd0aDsgaSsrKSB7XG4gICAgICBpZiAoaSA8IGlEaW0pIHtcbiAgICAgICAgcGF5bG9hZC5jb2x1bW5OYW1lcy5pbnB1dC5wdXNoKHhtbVRyYWluaW5nU2V0LmNvbHVtbl9uYW1lc1tpXSk7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBwYXlsb2FkLmNvbHVtbk5hbWVzLm91dHB1dC5wdXNoKHhtbVRyYWluaW5nU2V0LmNvbHVtbl9uYW1lc1tpXSk7XG4gICAgICB9XG4gICAgfVxuXG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBwaHJhc2VzLmxlbmd0aDsgaSsrKSB7XG4gICAgICBjb25zdCBleGFtcGxlID0ge1xuICAgICAgICBpbnB1dDpbXSxcbiAgICAgICAgb3V0cHV0OiBbXSxcbiAgICAgICAgbGFiZWw6IHBocmFzZXNbaV0ubGFiZWxcbiAgICAgIH07XG5cbiAgICAgIGZvciAobGV0IGogPSAwOyBqIDwgcGhyYXNlc1tpXS5sZW5ndGg7IGorKykge1xuICAgICAgICBleGFtcGxlLmlucHV0LnB1c2gocGhyYXNlc1tpXS5kYXRhX2lucHV0LnNsaWNlKGogKiBpRGltLCAoaiArIDEpICogaURpbSkpO1xuICAgICAgICBleGFtcGxlLm91dHB1dC5wdXNoKHBocmFzZXNbaV0uZGF0YV9vdXRwdXQuc2xpY2UoaiAqIG9EaW0sIChqICsgMSkgKiBvRGltKSk7XG4gICAgICB9XG5cbiAgICAgIHBheWxvYWQuZGF0YS5wdXNoKGV4YW1wbGUpO1xuICAgIH1cbiAgfSBlbHNlIHtcbiAgICBwYXlsb2FkLmlucHV0RGltZW5zaW9uID0geG1tVHJhaW5pbmdTZXQuZGltZW5zaW9uO1xuICAgIHBheWxvYWQub3V0cHV0RGltZW5zaW9uID0gMDtcblxuICAgIGNvbnN0IGRpbSA9IHBheWxvYWQuaW5wdXREaW1lbnNpb247XG5cbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHhtbVRyYWluaW5nU2V0LmNvbHVtbl9uYW1lcy5sZW5ndGg7IGkrKykge1xuICAgICAgcGF5bG9hZC5jb2x1bW5OYW1lcy5pbnB1dC5wdXNoKHhtbVRyYWluaW5nU2V0LmNvbHVtbl9uYW1lc1tpXSk7XG4gICAgfVxuXG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBwaHJhc2VzLmxlbmd0aDsgaSsrKSB7XG4gICAgICBjb25zdCBleGFtcGxlID0ge1xuICAgICAgICBpbnB1dDpbXSxcbiAgICAgICAgb3V0cHV0OiBbXSxcbiAgICAgICAgbGFiZWw6IHBocmFzZXNbaV0ubGFiZWxcbiAgICAgIH07XG5cbiAgICAgIGZvciAobGV0IGogPSAwOyBqIDwgcGhyYXNlc1tpXS5sZW5ndGg7IGorKykge1xuICAgICAgICBleGFtcGxlLmlucHV0LnB1c2gocGhyYXNlc1tpXS5kYXRhLnNsaWNlKGogKiBkaW0sIChqICsgMSkgKiBkaW0pKTtcbiAgICAgIH1cblxuICAgICAgcGF5bG9hZC5kYXRhLnB1c2goZXhhbXBsZSk7XG4gICAgfVxuICB9XG5cbiAgcmV0dXJuIHtcbiAgICBkb2NUeXBlOiAncmFwaWQtbWl4Om1sLXRyYWluaW5nLXNldCcsXG4gICAgZG9jVmVyc2lvbjogUkFQSURfTUlYX0RPQ19WRVJTSU9OLFxuICAgIHBheWxvYWQ6IHBheWxvYWQsXG4gIH07XG59XG5cbi8qKlxuICogQ29udmVydCBhIFJBUElELU1JWCBjb25maWd1cmF0aW9uIE9iamVjdCB0byBhbiBYTU0gY29uZmlndXJhdGlvbiBPYmplY3QuXG4gKlxuICogQHBhcmFtIHtPYmplY3R9IHJhcGlkTWl4Q29uZmlnIC0gQSBSQVBJRC1NSVggY29tcGF0aWJsZSBjb25maWd1cmFpdG9uIG9iamVjdFxuICpcbiAqIEByZXR1cm4ge09iamVjdH0geG1tQ29uZmlnIC0gQSBjb25maWd1cmF0aW9uIG9iamVjdCByZWFkeSB0byBiZSB1c2VkIGJ5IHRoZSBYTU0gbGlicmFyeVxuICovXG5jb25zdCByYXBpZE1peFRvWG1tQ29uZmlnID0gcmFwaWRNaXhDb25maWcgPT4ge1xuICByZXR1cm4gcmFwaWRNaXhDb25maWcucGF5bG9hZDtcbn1cblxuLyoqXG4gKiBDb252ZXJ0IGFuIFhNTSBjb25maWd1cmF0aW9uIE9iamVjdCB0byBhIFJBUElELU1JWCBjb25maWd1cmF0aW9uIHNldCBPYmplY3QuXG4gKlxuICogQHBhcmFtIHtPYmplY3R9IHhtbUNvbmZpZyAtIEEgY29uZmlndXJhdGlvbiBvYmplY3QgdGFyZ2V0aW5nIHRoZSBYTU0gbGlicmFyeVxuICpcbiAqIEByZXR1cm4ge09iamVjdH0gcmFwaWRNaXhDb25maWcgLSBBIFJBUElELU1JWCBjb21wYXRpYmxlIGNvbmZpZ3VyYXRpb24gb2JqZWN0XG4gKi9cbmNvbnN0IHhtbVRvUmFwaWRNaXhDb25maWcgPSB4bW1Db25maWcgPT4ge1xuICByZXR1cm4ge1xuICAgIGRvY1R5cGU6ICdyYXBpZC1taXg6bWwtY29uZmlndXJhdGlvbicsXG4gICAgZG9jVmVyc2lvbjogUkFQSURfTUlYX0RPQ19WRVJTSU9OLFxuICAgIHRhcmdldDoge1xuICAgICAgbmFtZTogYHhtbWAsXG4gICAgICB2ZXJzaW9uOiAnMS4wLjAnLFxuICAgIH0sXG4gICAgcGF5bG9hZDogeG1tQ29uZmlnLFxuICB9XG59XG5cbi8qKlxuICogQ29udmVydCBhIFJBUElELU1JWCBjb25maWd1cmF0aW9uIE9iamVjdCB0byBhbiBYTU0gY29uZmlndXJhdGlvbiBPYmplY3QuXG4gKlxuICogQHBhcmFtIHtPYmplY3R9IHJhcGlkTWl4TW9kZWwgLSBBIFJBUElELU1JWCBjb21wYXRpYmxlIG1vZGVsXG4gKlxuICogQHJldHVybiB7T2JqZWN0fSB4bW1Nb2RlbCAtIEEgbW9kZWwgcmVhZHkgdG8gYmUgdXNlZCBieSB0aGUgWE1NIGxpYnJhcnlcbiAqL1xuY29uc3QgcmFwaWRNaXhUb1htbU1vZGVsID0gcmFwaWRNaXhNb2RlbCA9PiB7XG4gIHJldHVybiByYXBpZE1peE1vZGVsLnBheWxvYWQ7XG59XG5cbi8qKlxuICogQ29udmVydCBhbiBYTU0gbW9kZWwgT2JqZWN0IHRvIGEgUkFQSUQtTUlYIG1vZGVsIE9iamVjdC5cbiAqXG4gKiBAcGFyYW0ge09iamVjdH0geG1tTW9kZWwgLSBBIG1vZGVsIGdlbmVyYXRlZCBieSB0aGUgWE1NIGxpYnJhcnlcbiAqXG4gKiBAcmV0dXJuIHtPYmplY3R9IHJhcGlkTWl4TW9kZWwgLSBBIFJBUElELU1JWCBjb21wYXRpYmxlIG1vZGVsXG4gKi9cbmNvbnN0IHhtbVRvUmFwaWRNaXhNb2RlbCA9IHhtbU1vZGVsID0+IHtcbiAgY29uc3QgbW9kZWxUeXBlID0geG1tTW9kZWwuY29uZmlndXJhdGlvbi5kZWZhdWx0X3BhcmFtZXRlcnMuc3RhdGVzID8gJ2hobW0nIDogJ2dtbSc7XG5cbiAgcmV0dXJuIHtcbiAgICBkb2NUeXBlOiAncmFwaWQtbWl4Om1sLW1vZGVsJyxcbiAgICBkb2NWZXJzaW9uOiBSQVBJRF9NSVhfRE9DX1ZFUlNJT04sXG4gICAgdGFyZ2V0OiB7XG4gICAgICBuYW1lOiBgeG1tYCxcbiAgICAgIHZlcnNpb246ICcxLjAuMCcsXG4gICAgfSxcbiAgICBwYXlsb2FkOiBPYmplY3QuYXNzaWduKHt9LCB4bW1Nb2RlbCwgeyBtb2RlbFR5cGUgfSksXG4gIH1cbn07XG5cbi8qXG4gKiBAbW9kdWxlIGNvbW9cbiAqXG4gKiBAZGVzY3JpcHRpb24gRm9yIHRoZSBtb21lbnQgdGhlIGNvbW8gd2ViIHNlcnZpY2Ugd2lsbCBvbmx5IHJldHVybiBYTU0gbW9kZWxzXG4gKiB3cmFwcGVkIGludG8gUkFQSUQtTUlYIEpTT04gb2JqZWN0cywgdGFraW5nIFJBUElELU1JWCB0cmFpbmluZ3Mgc2V0cyBhbmQgWE1NIGNvbmZpZ3VyYXRpb25zLlxuICovXG5cbi8qKlxuICogQ3JlYXRlIHRoZSBKU09OIHRvIHNlbmQgdG8gdGhlIENvbW8gd2ViIHNlcnZpY2UgdmlhIGh0dHAgcmVxdWVzdC5cbiAqXG4gKiBAcGFyYW0ge09iamVjdH0gY29uZmlnIC0gQSB2YWxpZCBSQVBJRC1NSVggY29uZmlndXJhdGlvbiBvYmplY3RcbiAqIEBwYXJhbSB7T2JqZWN0fSB0cmFpbmluZ1NldCAtIEEgdmFsaWQgUkFQSUQtTUlYIHRyYWluaW5nIHNldCBvYmplY3RcbiAqIEBwYXJhbSB7T2JqZWN0fSBbbWV0YXM9bnVsbF0gLSBTb21lIG9wdGlvbmFsIG1ldGEgZGF0YVxuICogQHBhcmFtIHtPYmplY3R9IFtzaWduYWxQcm9jZXNzaW5nPW51bGxdIC0gQW4gb3B0aW9uYWwgZGVzY3JpcHRpb24gb2YgdGhlIHByZSBwcm9jZXNzaW5nIHVzZWQgdG8gb2J0YWluIHRoZSB0cmFpbmluZyBzZXRcbiAqXG4gKiBAcmV0dXJuIHtPYmplY3R9IGh0dHBSZXF1ZXN0IC0gQSB2YWxpZCBKU09OIHRvIGJlIHNlbnQgdG8gdGhlIENvbW8gd2ViIHNlcnZpY2UgdmlhIGh0dHAgcmVxdWVzdC5cbiAqL1xuY29uc3QgY3JlYXRlQ29tb0h0dHBSZXF1ZXN0ID0gKGNvbmZpZywgdHJhaW5pbmdTZXQsIG1ldGFzID0gbnVsbCwgc2lnbmFsUHJvY2Vzc2luZyA9IG51bGwpID0+IHtcbiAgY29uc3QgcmVzcXVlc3QgPSB7XG4gICAgZG9jVHlwZTogJ3JhcGlkLW1peDptbC1odHRwLXJlcXVlc3QnLFxuICAgIGRvY1ZlcnNpb246IFJBUElEX01JWF9ET0NfVkVSU0lPTixcbiAgICB0YXJnZXQ6IHtcbiAgICAgIG5hbWU6ICdjb21vLXdlYi1zZXJ2aWNlJyxcbiAgICAgIHZlcnNpb246ICcxLjAuMCdcbiAgICB9LFxuICAgIHBheWxvYWQ6IHtcbiAgICAgIGNvbmZpZ3VyYXRpb246IGNvbmZpZyxcbiAgICAgIHRyYWluaW5nU2V0OiB0cmFpbmluZ1NldFxuICAgIH1cbiAgfTtcblxuICBpZiAobWV0YXMgIT09IG51bGwpIHtcbiAgICByZXNxdWVzdC5wYXlsb2FkLm1ldGFzID0gbWV0YXM7XG4gIH1cblxuICBpZiAoc2lnbmFsUHJvY2Vzc2luZyAhPT0gbnVsbCkge1xuICAgIHJlc3F1ZXN0LnBheWxvYWQuc2lnbmFsUHJvY2Vzc2luZyA9IHNpZ25hbFByb2Nlc3Npbmc7XG4gIH1cblxuICByZXR1cm4gcmVzcXVlc3Q7XG59O1xuXG4vKipcbiAqIENyZWF0ZSB0aGUgSlNPTiB0byBzZW5kIGJhY2sgYXMgYSByZXNwb25zZSB0byBodHRwIHJlcXVlc3RzIHRvIHRoZSBDb21vIHdlYiBzZXJ2aWNlLlxuICpcbiAqIEBwYXJhbSB7T2JqZWN0fSBjb25maWcgLSBBIHZhbGlkIFJBUElELU1JWCBjb25maWd1cmF0aW9uIG9iamVjdFxuICogQHBhcmFtIHtPYmplY3R9IG1vZGVsIC0gQSB2YWxpZCBSQVBJRC1NSVggbW9kZWwgb2JqZWN0XG4gKiBAcGFyYW0ge09iamVjdH0gW21ldGFzPW51bGxdIC0gU29tZSBvcHRpb25hbCBtZXRhIGRhdGFcbiAqIEBwYXJhbSB7T2JqZWN0fSBbc2lnbmFsUHJvY2Vzc2luZz1udWxsXSAtIEFuIG9wdGlvbmFsIGRlc2NyaXB0aW9uIG9mIHRoZSBwcmUgcHJvY2Vzc2luZyB1c2VkIHRvIG9idGFpbiB0aGUgdHJhaW5pbmcgc2V0XG4gKlxuICogQHJldHVybiB7T2JqZWN0fSBodHRwUmVzcG9uc2UgLSBBIHZhbGlkIEpTT04gcmVzcG9uc2UgdG8gYmUgc2VudCBiYWNrIGZyb20gdGhlIENvbW8gd2ViIHNlcnZpY2UgdmlhIGh0dHAuXG4gKi9cbmNvbnN0IGNyZWF0ZUNvbW9IdHRwUmVzcG9uc2UgPSAoY29uZmlnLCBtb2RlbCwgbWV0YXMgPSBudWxsLCBzaWduYWxQcm9jZXNzaW5nID0gbnVsbCkgPT4ge1xuICBjb25zdCByZXNwb25zZSA9IHtcbiAgICBkb2NUeXBlOiAncmFwaWQtbWl4Om1sLWh0dHAtcmVzcG9uc2UnLFxuICAgIGRvY1ZlcnNpb246IFJBUElEX01JWF9ET0NfVkVSU0lPTixcbiAgICB0YXJnZXQ6IHtcbiAgICAgIG5hbWU6ICdjb21vLXdlYi1zZXJ2aWNlJyxcbiAgICAgIHZlcnNpb246ICcxLjAuMCdcbiAgICB9LFxuICAgIHBheWxvYWQ6IHtcbiAgICAgIGNvbmZpZ3VyYXRpb246IGNvbmZpZyxcbiAgICAgIG1vZGVsOiBtb2RlbFxuICAgIH1cbiAgfTtcblxuICBpZiAobWV0YXMgIT09IG51bGwpIHtcbiAgICByZXNwb25zZS5wYXlsb2FkLm1ldGFzID0gbWV0YXM7XG4gIH1cblxuICBpZiAoc2lnbmFsUHJvY2Vzc2luZyAhPT0gbnVsbCkge1xuICAgIHJlc3BvbnNlLnBheWxvYWQuc2lnbmFsUHJvY2Vzc2luZyA9IHNpZ25hbFByb2Nlc3Npbmc7XG4gIH1cblxuICByZXR1cm4gcmVzcG9uc2U7XG59O1xuXG5cbmV4cG9ydCBkZWZhdWx0IHtcbiAgLy8gcmFwaWRMaWIgYWRhcHRlcnNcbiAgcmFwaWRNaXhUb1JhcGlkTGliVHJhaW5pbmdTZXQsXG5cbiAgLy8geG1tIGFkYXB0ZXJzXG4gIHJhcGlkTWl4VG9YbW1UcmFpbmluZ1NldCxcbiAgeG1tVG9SYXBpZE1peFRyYWluaW5nU2V0LFxuXG4gIHJhcGlkTWl4VG9YbW1Db25maWcsXG4gIHhtbVRvUmFwaWRNaXhDb25maWcsXG5cbiAgcmFwaWRNaXhUb1htbU1vZGVsLFxuICB4bW1Ub1JhcGlkTWl4TW9kZWwsXG5cbiAgY3JlYXRlQ29tb0h0dHBSZXF1ZXN0LFxuICBjcmVhdGVDb21vSHR0cFJlc3BvbnNlLFxuICAvLyBjb25zdGFudHNcbiAgUkFQSURfTUlYX0RPQ19WRVJTSU9OLFxuICBSQVBJRF9NSVhfREVGQVVMVF9MQUJFTFxufTtcbiJdfQ==