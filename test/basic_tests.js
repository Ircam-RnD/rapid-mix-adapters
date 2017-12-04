import rapidMixAdapters from '../src';
import xmmClient from 'xmm-client';
import test from 'tape';

test('basic translation tests', (t) => {

  /* * * * * * * * * * * * * * * TRAINING SET * * * * * * * * * * * * * * * * */

  const rmSet = {
    docType: 'rapid-mix:ml-training-set',
    docVersion: '1.0.0',
    payload: {
      inputDimension: 3,
      outputDimension: 1,
      columnNames: { // optional
        input: [ 'accelX', 'accelY', 'accelZ' ],
        output: [ 'volume' ] // optional
      },
      data: [
        {
          input: [
            [1, 2, 3], // frame 0
            [1.1, 1.9, 2.5], // frame 1
            [1.5, 1.5, 2]
          ],
          output: [ // can be a single empty array if outputDimension is 0
            [ 0 ], [ 0.1 ], [ 0.2 ]
          ],
          label: 'label1', // optional
        },
      ]
    }
  };

  const xSet = rapidMixAdapters.rapidMixToXmmTrainingSet(rmSet);
  const rmSet2 = rapidMixAdapters.xmmToRapidMixTrainingSet(xSet);
  const xSet2 = rapidMixAdapters.rapidMixToXmmTrainingSet(rmSet2);

  // console.log(JSON.stringify(xSet, null, 2));

  t.deepEqual(rmSet, rmSet2, 'translation should work back and forth');
  t.deepEqual(xSet, xSet2, 'translation should work back and forth');

  // const p = new xmmClient.PhraseMaker();
  // const s = new xmmClient.SetMaker();

  const xConfig = {
    modelType: 'hhmm',
    gaussians: 3
  };

  /* * * * * * * * * * * * * * * * CONFIG * * * * * * * * * * * * * * * * * * */

  // const rmConfig = rapidMixAdapters.xmmToRapidMixConfig(xConfig);
  // const xConfig2 = rapidMixAdapters.rapidMixToXmmConfig(rmConfig);

  // console.log(JSON.stringify(rmConfig, null, 2));

  // t.deepEqual(xConfig, xConfig2, 'config should be translated');

  t.end();
});
