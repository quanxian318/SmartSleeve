/**
 * RandomForest inference engine for WeChat Mini Program.
 *
 * Binary format (little-endian, 13 + n_outputs*4 bytes per node):
 *   Header: n_estimators(u32) n_features(u32) n_outputs(u32)
 *   Per tree: n_nodes(u32) + children_left(i32[n]) + children_right(i32[n])
 *             + feature(u8[n]) + threshold(f32[n]) + values(f32[n_outputs*n])
 *
 * All TypedArrays use buffer.slice() to create independent ArrayBuffers,
 * avoiding platform alignment issues on iOS/Android.
 */

const rfInference = {
  _trees: null,
  _nEstimators: 0,
  _nFeatures: 0,
  _nOutputs: 0,
  _loaded: false,

  loadModel(buffer) {
    var dv = new DataView(buffer);
    var bufLen = buffer.byteLength;
    console.log('[rfInference] Parsing buffer: ' + bufLen + ' bytes');

    var offset = 0;
    this._nEstimators = dv.getUint32(offset, true); offset += 4;
    this._nFeatures   = dv.getUint32(offset, true); offset += 4;
    this._nOutputs    = dv.getUint32(offset, true); offset += 4;

    console.log('[rfInference] Header: n_est=' + this._nEstimators +
      ' n_feat=' + this._nFeatures + ' n_out=' + this._nOutputs);

    if (this._nEstimators < 1 || this._nEstimators > 10000 ||
        this._nFeatures < 1 || this._nFeatures > 1000 ||
        this._nOutputs < 1 || this._nOutputs > 100) {
      throw new Error('Invalid header — file may be corrupted or still compressed');
    }

    // Bytes per node: children_left(4) + children_right(4) + feature(1) + threshold(4) + values(n_outputs*4)
    var nodeBytes = 13 + this._nOutputs * 4;

    this._trees = [];

    for (var t = 0; t < this._nEstimators; t++) {
      if (offset + 4 > bufLen) {
        throw new Error('Unexpected end of file at tree ' + t);
      }

      var nNodes = dv.getUint32(offset, true);

      if (nNodes < 1 || nNodes > 100000) {
        throw new Error('Invalid nNodes=' + nNodes + ' at tree ' + t +
          ' (offset=' + offset + ', bufLen=' + bufLen + ')');
      }

      var treeEnd = offset + 4 + nNodes * nodeBytes;
      if (treeEnd > bufLen) {
        throw new Error('Tree ' + t + ' exceeds buffer: need ' + treeEnd +
          ' bytes, have ' + bufLen + ' (nNodes=' + nNodes + ')');
      }

      // slice() creates independent ArrayBuffers at offset 0 — always aligned
      var o = offset + 4; // skip n_nodes
      var childrenLeft  = new Int32Array(buffer.slice(o, o + nNodes * 4));
      o += nNodes * 4;
      var childrenRight = new Int32Array(buffer.slice(o, o + nNodes * 4));
      o += nNodes * 4;
      var feature       = new Uint8Array(buffer.slice(o, o + nNodes));
      o += nNodes;
      var threshold     = new Float32Array(buffer.slice(o, o + nNodes * 4));
      o += nNodes * 4;
      var values        = new Float32Array(buffer.slice(o, o + nNodes * this._nOutputs * 4));

      this._trees.push({
        childrenLeft:  childrenLeft,
        childrenRight: childrenRight,
        feature:       feature,
        threshold:     threshold,
        values:        values,
        nNodes: nNodes
      });

      offset = treeEnd;
    }

    if (offset !== bufLen) {
      console.warn('[rfInference] Extra ' + (bufLen - offset) +
        ' bytes after last tree (expected exact match)');
    }

    this._loaded = true;
    console.log('[rfInference] OK: ' + this._nEstimators + ' trees loaded');
  },

  predict(features) {
    if (!this._loaded) throw new Error('Model not loaded');

    var nOut = this._nOutputs;
    var sums = new Array(nOut);
    for (var k = 0; k < nOut; k++) sums[k] = 0;

    var trees = this._trees;

    for (var t = 0; t < trees.length; t++) {
      var childrenLeft = trees[t].childrenLeft;
      var childrenRight = trees[t].childrenRight;
      var feature = trees[t].feature;
      var threshold = trees[t].threshold;
      var values = trees[t].values;
      var node = 0;

      while (childrenLeft[node] !== -1) {
        if (features[feature[node]] <= threshold[node]) {
          node = childrenLeft[node];
        } else {
          node = childrenRight[node];
        }
      }

      var vi = node * nOut;
      for (var k = 0; k < nOut; k++) {
        sums[k] += values[vi + k];
      }
    }

    var invN = 1.0 / trees.length;
    var result = new Array(nOut);
    for (var k = 0; k < nOut; k++) {
      result[k] = sums[k] * invN;
    }
    return result;
  },

  /**
   * Convenience: return named object { brachioradialis, biceps, triceps }.
   */
  predictNamed: function (features) {
    var raw = this.predict(features);
    return {
      brachioradialis: raw[0],
      biceps: raw[1],
      triceps: raw[2]
    };
  },

  isLoaded: function () {
    return this._loaded;
  }
};

module.exports = rfInference;
