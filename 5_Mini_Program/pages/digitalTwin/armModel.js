/**
 * Arm model loader using Z-Anatomy GLB from Blender.
 *
 * Loads a pre-built glTF binary with pivot hierarchy:
 *   armRoot → shoulderPivot → [humerus, biceps, triceps, deltoid, ...]
 *                            → elbowPivot → [radius, ulna, brachioradialis, ...]
 *                                          → wristPivot → [hand bones]
 *
 * Requires gltfLoader.js (adapted GLTFLoader for THREE r108).
 */

var gltfLoaderFactory = require('./gltfLoader');

function loadArmModel(buffer) {
  var THREE = require('./threeAdapter').THREE;
  if (!THREE) {
    return Promise.reject(new Error('THREE not initialized'));
  }

  var GLTFLoader = gltfLoaderFactory(THREE);
  var loader = new GLTFLoader();

  return new Promise(function (resolve, reject) {
    loader.parse(buffer, '', function (gltf) {
      try {
        var result = setupModel(THREE, gltf);
        resolve(result);
      } catch (e) {
        reject(e);
      }
    }, function (err) {
      reject(err);
    });
  });
}

function setupModel(THREE, gltf) {
  var scene = gltf.scene;

  // The GLB stores WORLD-SPACE positions in glTF translation fields.
  // GLTFLoader interprets them as LOCAL, which means each node's .position
  // is its world position from the Blender export.
  // Strategy: read .position as world position, rebuild clean hierarchy.

  // Find the original pivot nodes
  var origArmRoot = scene.getObjectByName('armRoot');
  var origShoulder = scene.getObjectByName('shoulderPivot');
  var origElbow = scene.getObjectByName('elbowPivot');
  var origWrist = scene.getObjectByName('wristPivot');

  if (!origShoulder) {
    console.error('[armModel] shoulderPivot not found');
    return null;
  }

  // Pivot world positions = their .position values (glTF translations)
  var shoulderWorld = origShoulder.position.clone();
  var elbowWorld = origElbow ? origElbow.position.clone() : new THREE.Vector3(0, -1.5, 0);
  var wristWorld = origWrist ? origWrist.position.clone() : new THREE.Vector3(0, -2.5, 0);

  console.log('[armModel] shoulder pos:', shoulderWorld.x.toFixed(3), shoulderWorld.y.toFixed(3), shoulderWorld.z.toFixed(3));
  console.log('[armModel] elbow pos:', elbowWorld.x.toFixed(3), elbowWorld.y.toFixed(3), elbowWorld.z.toFixed(3));
  console.log('[armModel] wrist pos:', wristWorld.x.toFixed(3), wristWorld.y.toFixed(3), wristWorld.z.toFixed(3));

  // --- Create new clean pivot hierarchy ---
  var armRoot = new THREE.Group();
  armRoot.name = 'armRoot';

  var shoulderPivot = new THREE.Group();
  shoulderPivot.name = 'shoulderPivot';
  shoulderPivot.position.copy(shoulderWorld);
  armRoot.add(shoulderPivot);

  var elbowPivot = new THREE.Group();
  elbowPivot.name = 'elbowPivot';
  elbowPivot.position.copy(elbowWorld).sub(shoulderWorld);
  shoulderPivot.add(elbowPivot);

  console.log('[armModel] elbow local:', elbowPivot.position.x.toFixed(3), elbowPivot.position.y.toFixed(3), elbowPivot.position.z.toFixed(3));

  // Add wristPivot
  var wristPivot = new THREE.Group();
  wristPivot.name = 'wristPivot';
  wristPivot.position.copy(wristWorld).sub(elbowWorld);
  elbowPivot.add(wristPivot);

  console.log('[armModel] wrist local:', wristPivot.position.x.toFixed(3), wristPivot.position.y.toFixed(3), wristPivot.position.z.toFixed(3));

  // --- Reparent meshes from old hierarchy to new pivots ---
  var materials = {};
  var muscles = {};

  function reparentChildren(fromPivot, toPivot, toPivotWorld) {
    if (!fromPivot) return;
    var kids = fromPivot.children.slice();
    kids.forEach(function (child) {
      fromPivot.remove(child);
      child.position.sub(toPivotWorld);
      toPivot.add(child);
    });
  }

  reparentChildren(origShoulder, shoulderPivot, shoulderWorld);
  reparentChildren(origElbow, elbowPivot, elbowWorld);
  reparentChildren(origWrist, wristPivot, wristWorld);
  reparentChildren(origArmRoot, armRoot, new THREE.Vector3(0, 0, 0));

  console.log('[armModel] Reparenting complete');

  // --- Scale up ---
  armRoot.scale.set(8, 8, 8);

  // --- Find and set up muscle meshes ---
  function findNode(name) {
    var found = armRoot.getObjectByName(name);
    return found;
  }

  var bicepsNode = findNode('biceps');
  var tricepsNode = findNode('triceps');
  var deltoidNode = findNode('deltoid');
  var brachioradNode = findNode('brachioradialis');

  console.log('[armModel] Found meshes:',
    'biceps=' + !!bicepsNode,
    'triceps=' + !!tricepsNode,
    'deltoid=' + !!deltoidNode,
    'brachioradialis=' + !!brachioradNode
  );

  function setupMuscleMesh(node, key) {
    if (!node) return;

    var meshList = [];
    if (node.isMesh) meshList.push(node);
    node.traverse(function (child) {
      if (child.isMesh && child !== node) meshList.push(child);
    });

    node.userData = node.userData || {};
    node.userData.muscle = key;

    var mat = null;
    if (meshList.length > 0) {
      var first = meshList[0];
      if (first.material) {
        mat = Array.isArray(first.material)
          ? first.material[0].clone()
          : first.material.clone();
      }
    }

    if (!mat) {
      mat = new THREE.MeshStandardMaterial({
        color: 0xc4594a, roughness: 0.55, metalness: 0.02,
        transparent: true, opacity: 0.8
      });
      console.warn('[armModel] No material for ' + key + ', using default');
    }

    meshList.forEach(function (m) { m.material = mat; });
    materials[key] = mat;
    muscles[key] = node;
    console.log('[armModel] ' + key + ': ' + meshList.length + ' mesh(es)');
  }

  setupMuscleMesh(bicepsNode, 'biceps');
  setupMuscleMesh(tricepsNode, 'triceps');
  setupMuscleMesh(deltoidNode, 'deltoid');
  setupMuscleMesh(brachioradNode, 'brachioradialis');

  // Fallback for partial meshes
  if (!muscles.biceps) {
    var longHead = armRoot.getObjectByName('Long head of biceps brachii.r');
    if (longHead) {
      longHead.userData = { muscle: 'biceps', displayName: '肱二头肌', enName: 'Biceps Brachii' };
      muscles.biceps = longHead;
    }
  }

  // Display names
  var names = {
    biceps: { displayName: '肱二头肌', enName: 'Biceps Brachii' },
    triceps: { displayName: '肱三头肌', enName: 'Triceps Brachii' },
    deltoid: { displayName: '三角肌', enName: 'Deltoid' },
    brachioradialis: { displayName: '肱桡肌', enName: 'Brachioradialis' }
  };

  Object.keys(muscles).forEach(function (key) {
    if (muscles[key] && muscles[key].userData && names[key]) {
      muscles[key].userData.displayName = names[key].displayName;
      muscles[key].userData.enName = names[key].enName;
    }
  });

  // Bone materials — all skeletal parts get ivory tones
  var BONE_COLOR_MAIN = 0xe8d5c0;      // humerus/radius/ulna
  var BONE_COLOR_GIRDLE = 0xd5c8b5;    // scapula/clavicle
  var BONE_COLOR_HAND = 0xf0e8d8;      // carpals/metacarpals/phalanges
  var OTHER_MUSCLE_COLOR = 0x6b8fa3;   // slate blue-gray, distinct from EMG warm colors

  function colorMeshNode(node, colorHex, roughness, metalness) {
    var list = [];
    if (node.isMesh) list.push(node);
    node.traverse(function (c) { if (c.isMesh && c !== node) list.push(c); });
    list.forEach(function (m) {
      if (m.material) {
        m.material.color.set(colorHex);
        m.material.roughness = roughness;
        m.material.metalness = metalness;
        m.material.emissive = m.material.emissive || new THREE.Color(0x000000);
        m.material.emissive.set(0x000000);
        m.material.emissiveIntensity = 0;
      }
    });
  }

  function isHandBone(name) {
    return name.indexOf('carpal') !== -1 || name.indexOf('metacarpal') !== -1 ||
      name.indexOf('phalanx') !== -1 || name.indexOf('Scaphoid') !== -1 ||
      name.indexOf('Lunate') !== -1 || name.indexOf('Triquetrum') !== -1 ||
      name.indexOf('Pisiform') !== -1 || name.indexOf('Trapezium') !== -1 ||
      name.indexOf('Trapezoid') !== -1 || name.indexOf('Capitate') !== -1 ||
      name.indexOf('Hamate') !== -1;
  }

  // Color main long bones
  ['humerus', 'radius', 'ulna'].forEach(function (name) {
    var bone = armRoot.getObjectByName(name);
    if (bone) colorMeshNode(bone, BONE_COLOR_MAIN, 0.4, 0.02);
  });

  // Color shoulder girdle bones
  ['Scapula.r', 'Clavicle.r'].forEach(function (name) {
    var bone = armRoot.getObjectByName(name);
    if (bone) colorMeshNode(bone, BONE_COLOR_GIRDLE, 0.45, 0.03);
  });

  // Collect the 4 interactive muscle names so we skip them
  var muscleKeys = Object.keys(muscles);

  // Color ALL remaining meshes that are not the 4 interactive muscles
  armRoot.traverse(function (obj) {
    if (!obj.isMesh) return;
    // Skip if already handled by muscle setup
    var cur = obj;
    var belongsToMuscle = false;
    while (cur) {
      if (cur.userData && cur.userData.muscle && muscleKeys.indexOf(cur.userData.muscle) !== -1) {
        belongsToMuscle = true;
        break;
      }
      cur = cur.parent;
    }
    if (belongsToMuscle) return;

    var name = obj.name || '';
    // Skip if this mesh is already using a muscle material (check parent chain)
    if (obj.material) {
      var isMuscleMat = false;
      for (var mk = 0; mk < muscleKeys.length; mk++) {
        if (obj.material === materials[muscleKeys[mk]]) {
          isMuscleMat = true;
          break;
        }
      }
      if (isMuscleMat) return;

      if (isHandBone(name)) {
        obj.material.color.set(BONE_COLOR_HAND);
        obj.material.roughness = 0.42;
        obj.material.metalness = 0.02;
        obj.material.emissive = obj.material.emissive || new THREE.Color(0x000000);
        obj.material.emissive.set(0x000000);
        obj.material.emissiveIntensity = 0;
      } else if (name.indexOf('Scapula') !== -1 || name.indexOf('Clavicle') !== -1) {
        // Already handled above via getObjectByName
      } else if (name === 'humerus' || name === 'radius' || name === 'ulna') {
        // Already handled above
      } else {
        // All other tissues (other muscles, tendons, etc.) → slate blue-gray
        obj.material.color.set(OTHER_MUSCLE_COLOR);
        obj.material.roughness = 0.55;
        obj.material.metalness = 0.03;
        obj.material.emissive = obj.material.emissive || new THREE.Color(0x000000);
        obj.material.emissive.set(0x000000);
        obj.material.emissiveIntensity = 0;
        obj.material.transparent = true;
        obj.material.opacity = 0.75;
      }
    }
  });

  // Compute model bounds for camera positioning
  armRoot.updateMatrixWorld(true);
  var box = new THREE.Box3();
  armRoot.traverse(function (obj) {
    if (obj.isMesh && obj.geometry) {
      obj.geometry.computeBoundingBox();
      if (obj.geometry.boundingBox) {
        var gBox = obj.geometry.boundingBox.clone();
        gBox.applyMatrix4(obj.matrixWorld);
        // r108: use union() instead of expandByBox/expandByObject
        box.union(gBox);
      }
    }
  });
  var hasBounds = (box.max.x - box.min.x) > 0.001;
  if (hasBounds) {
    var size = new THREE.Vector3();
    box.getSize(size);
    console.log('[armModel] Size:', size.x.toFixed(2), size.y.toFixed(2), size.z.toFixed(2));
    var center = new THREE.Vector3();
    box.getCenter(center);
    console.log('[armModel] Center (world):', center.x.toFixed(2), center.y.toFixed(2), center.z.toFixed(2));
  } else {
    console.log('[armModel] Using pivot-based size estimate');
  }

  return {
    rootGroup: armRoot,
    shoulderPivot: shoulderPivot,
    elbowPivot: elbowPivot,
    muscles: muscles,
    materials: materials
  };
}

module.exports = {
  loadArmModel: loadArmModel,
  buildArmModel: function () {
    console.warn('[armModel] buildArmModel is deprecated, use loadArmModel(buffer)');
    return null;
  }
};
