import { useState, useRef, useEffect, useCallback } from "react";
import * as THREE from "three";
import {
  Plus,
  Minus,
  RotateCw,
  RotateCcw,
  Shuffle,
  PartyPopper,
  Sparkles,
} from "lucide-react";

const AXIS = {
  x: { hex: "#FF6B6B", dark: "#C93E3E", label: "X" },
  y: { hex: "#51CF66", dark: "#2F9E44", label: "Y" },
  z: { hex: "#339AF0", dark: "#1C6FC4", label: "Z" },
};

const INK = "#2B2B2B";
const TRANS_STEP = 0.75;

// =========================
// ROTATION SETTINGS
// =========================

// Every button press = exactly 90°
const ROT_STEP_DEG = 90;
const ROT_STEP = THREE.MathUtils.degToRad(ROT_STEP_DEG);

// Animation duration for one 90° rotation
const ROT_ANIMATION_MS = 350;

const FLOOR_Y = -3.6;

// Random challenge rotations are always multiples of 90°.
const ALLOWED_ANGLES_DEG = [90, 180, 270];

function fmt(v) {
  let s = v.toFixed(2);
  if (s === "-0.00") s = "0.00";
  return s;
}

function randInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function randomChallenge() {
  let stepsX, stepsY, stepsZ;

  do {
    stepsX = randInt(-4, 4);
    stepsY = randInt(-4, 4);
    stepsZ = randInt(-4, 4);
  } while (
    Math.abs(stepsX) +
      Math.abs(stepsY) +
      Math.abs(stepsZ) <
    3
  );

  const pos = new THREE.Vector3(
    stepsX * TRANS_STEP,
    stepsY * TRANS_STEP,
    stepsZ * TRANS_STEP
  );

  const axesVecs = {
    x: new THREE.Vector3(1, 0, 0),
    y: new THREE.Vector3(0, 1, 0),
    z: new THREE.Vector3(0, 0, 1),
  };

  const quat = new THREE.Quaternion();

  // Generate a random orientation using
  // only multiples of 90°.
  const numOps = randInt(3, 5);

  for (let i = 0; i < numOps; i++) {
    const axisKey =
      ["x", "y", "z"][randInt(0, 2)];

    const angleDeg =
      ALLOWED_ANGLES_DEG[
        randInt(
          0,
          ALLOWED_ANGLES_DEG.length - 1
        )
      ];

    const sign =
      Math.random() < 0.5 ? -1 : 1;

    const dq =
      new THREE.Quaternion().setFromAxisAngle(
        axesVecs[axisKey],
        THREE.MathUtils.degToRad(angleDeg) * sign
      );

    quat.premultiply(dq);
  }

  return { pos, quat };
}

function buildToyArrow(dir, color, length) {
  const group = new THREE.Group();

  const rodLen = length * 0.72;
  const rodRadius = 0.09;
  const tipLen = length * 0.28;
  const tipRadius = 0.19;

  const quatDir =
    new THREE.Quaternion().setFromUnitVectors(
      new THREE.Vector3(0, 1, 0),
      dir
    );

  function makeMesh(
    geom,
    mat,
    scaleMul = 1
  ) {
    const mesh = new THREE.Mesh(geom, mat);
    mesh.scale.setScalar(scaleMul);
    return mesh;
  }

  const rodGeom =
    new THREE.CylinderGeometry(
      rodRadius,
      rodRadius,
      rodLen,
      14
    );

  const rodOutlineGeom =
    new THREE.CylinderGeometry(
      rodRadius,
      rodRadius,
      rodLen,
      14
    );

  const rodMat =
    new THREE.MeshBasicMaterial({ color });

  const rodOutlineMat =
    new THREE.MeshBasicMaterial({
      color: INK,
      side: THREE.BackSide,
    });

  const rodOutline = makeMesh(
    rodOutlineGeom,
    rodOutlineMat,
    1.22
  );

  const rod = makeMesh(
    rodGeom,
    rodMat
  );

  [rod, rodOutline].forEach((m) => {
    m.quaternion.copy(quatDir);

    m.position.copy(
      dir.clone().multiplyScalar(
        rodLen / 2
      )
    );
  });

  group.add(
    rodOutline,
    rod
  );

  const tipGeom =
    new THREE.ConeGeometry(
      tipRadius,
      tipLen,
      16
    );

  const tipOutlineGeom =
    new THREE.ConeGeometry(
      tipRadius,
      tipLen,
      16
    );

  const tipMat =
    new THREE.MeshBasicMaterial({
      color,
    });

  const tipOutlineMat =
    new THREE.MeshBasicMaterial({
      color: INK,
      side: THREE.BackSide,
    });

  const tipOutline = makeMesh(
    tipOutlineGeom,
    tipOutlineMat,
    1.18
  );

  const tip = makeMesh(
    tipGeom,
    tipMat
  );

  [tip, tipOutline].forEach((m) => {
    m.quaternion.copy(quatDir);

    m.position.copy(
      dir.clone().multiplyScalar(
        rodLen + tipLen / 2
      )
    );
  });

  group.add(
    tipOutline,
    tip
  );

  return group;
}

function buildFrame(length = 2.3) {
  const group = new THREE.Group();

  group.add(
    buildToyArrow(
      new THREE.Vector3(1, 0, 0),
      AXIS.x.hex,
      length
    )
  );

  group.add(
    buildToyArrow(
      new THREE.Vector3(0, 1, 0),
      AXIS.y.hex,
      length
    )
  );

  group.add(
    buildToyArrow(
      new THREE.Vector3(0, 0, 1),
      AXIS.z.hex,
      length
    )
  );

  return group;
}

function buildOriginBlob(color) {
  const g = new THREE.Group();

  const outline = new THREE.Mesh(
    new THREE.SphereGeometry(
      0.26,
      20,
      20
    ),
    new THREE.MeshBasicMaterial({
      color: INK,
      side: THREE.BackSide,
    })
  );

  outline.scale.setScalar(1.18);

  const ball = new THREE.Mesh(
    new THREE.SphereGeometry(
      0.26,
      20,
      20
    ),
    new THREE.MeshBasicMaterial({
      color,
    })
  );

  const highlight = new THREE.Mesh(
    new THREE.SphereGeometry(
      0.08,
      12,
      12
    ),
    new THREE.MeshBasicMaterial({
      color: 0xffffff,
    })
  );

  highlight.position.set(
    0.11,
    0.11,
    0.2
  );

  g.add(
    outline,
    ball,
    highlight
  );

  return g;
}

function buildShadowBlob() {
  const geom =
    new THREE.CircleGeometry(
      0.55,
      24
    );

  const mat =
    new THREE.MeshBasicMaterial({
      color: 0x2b2b2b,
      transparent: true,
      opacity: 0.14,
    });

  const mesh = new THREE.Mesh(
    geom,
    mat
  );

  mesh.rotation.x = -Math.PI / 2;

  return mesh;
}

export default function RotationVisualizer() {
  const mountRef = useRef(null);
  const groupRef = useRef(null);
  const shadowRef = useRef(null);

  // Used to cancel an active rotation animation.
  const rotationAnimationRef =
    useRef(null);

  const stateRef = useRef({
    pos: new THREE.Vector3(),
    quat: new THREE.Quaternion(),
  });

  const sphericalRef = useRef({
    theta: Math.PI / 4,
    phi: Math.PI / 2.6,
    radius: 14,
  });

  const dragRef = useRef({
    dragging: false,
    x: 0,
    y: 0,
  });

  const [matrixEls, setMatrixEls] =
    useState(Array(9).fill(0));

  const [translation, setTranslation] =
    useState([0, 0, 0]);

  const [posErr, setPosErr] =
    useState(0);

  const [angErr, setAngErr] =
    useState(0);

  const [aligned, setAligned] =
    useState(false);

  const [moves, setMoves] =
    useState(0);

  const [pop, setPop] =
    useState(false);

  const updateReadouts = useCallback(() => {
    const q = stateRef.current.quat;

    const m =
      new THREE.Matrix4()
        .makeRotationFromQuaternion(q);

    const e = m.elements;

    setMatrixEls([
      e[0],
      e[4],
      e[8],

      e[1],
      e[5],
      e[9],

      e[2],
      e[6],
      e[10],
    ]);

    const p =
      stateRef.current.pos;

    setTranslation([
      p.x,
      p.y,
      p.z,
    ]);

    const pe = p.length();

    const ae =
      THREE.MathUtils.radToDeg(
        2 *
          Math.acos(
            THREE.MathUtils.clamp(
              Math.abs(q.w),
              -1,
              1
            )
          )
      );

    setPosErr(pe);
    setAngErr(ae);

    const isAligned =
      pe < 0.06 &&
      ae < 3;

    setAligned((prev) => {
      if (
        isAligned &&
        !prev
      ) {
        setPop(true);

        setTimeout(() => {
          setPop(false);
        }, 900);
      }

      return isAligned;
    });
  }, []);

  useEffect(() => {
    const mount =
      mountRef.current;

    if (!mount) return;

    const width =
      mount.clientWidth || 320;

    const height =
      mount.clientHeight || 320;

    const scene =
      new THREE.Scene();

    const camera =
      new THREE.PerspectiveCamera(
        42,
        width / height,
        0.1,
        100
      );

    const renderer =
      new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
      });

    renderer.setSize(
      width,
      height
    );

    renderer.setPixelRatio(
      Math.min(
        window.devicePixelRatio,
        2
      )
    );

    renderer.domElement.style.display =
      "block";

    renderer.domElement.style.width =
      "100%";

    renderer.domElement.style.height =
      "100%";

    mount.appendChild(
      renderer.domElement
    );

    const floor = new THREE.Mesh(
      new THREE.CircleGeometry(
        9,
        40
      ),
      new THREE.MeshBasicMaterial({
        color: 0xfff3da,
        transparent: true,
        opacity: 0.9,
      })
    );

    floor.rotation.x =
      -Math.PI / 2;

    floor.position.y =
      FLOOR_Y;

    scene.add(floor);

    const ring = new THREE.Mesh(
      new THREE.RingGeometry(
        8.85,
        9,
        60
      ),
      new THREE.MeshBasicMaterial({
        color: INK,
        transparent: true,
        opacity: 0.15,
      })
    );

    ring.rotation.x =
      -Math.PI / 2;

    ring.position.y =
      FLOOR_Y + 0.001;

    scene.add(ring);

    for (
      let r = 2;
      r <= 8;
      r += 2
    ) {
      const dots =
        new THREE.RingGeometry(
          r - 0.02,
          r + 0.02,
          48
        );

      const dotMat =
        new THREE.MeshBasicMaterial({
          color: 0xe7c98f,
          transparent: true,
          opacity: 0.35,
        });

      const dotMesh =
        new THREE.Mesh(
          dots,
          dotMat
        );

      dotMesh.rotation.x =
        -Math.PI / 2;

      dotMesh.position.y =
        FLOOR_Y + 0.002;

      scene.add(dotMesh);
    }

    const fixedShadow =
      buildShadowBlob();

    fixedShadow.position.set(
      0,
      FLOOR_Y + 0.003,
      0
    );

    scene.add(
      fixedShadow
    );

    const movingShadow =
      buildShadowBlob();

    movingShadow.position.set(
      0,
      FLOOR_Y + 0.003,
      0
    );

    scene.add(
      movingShadow
    );

    shadowRef.current =
      movingShadow;

    const fixedFrame =
      buildFrame();

    scene.add(
      fixedFrame
    );

    scene.add(
      buildOriginBlob(
        "#FFD43B"
      )
    );

    const movingFrame =
      buildFrame();

    scene.add(
      movingFrame
    );

    groupRef.current =
      movingFrame;

    movingFrame.add(
      buildOriginBlob(
        "#63E6E8"
      )
    );

    const stringMat =
      new THREE.LineDashedMaterial({
        color: 0x2b2b2b,
        dashSize: 0.16,
        gapSize: 0.14,
        transparent: true,
        opacity: 0.35,
      });

    const stringGeom =
      new THREE.BufferGeometry()
        .setFromPoints([
          new THREE.Vector3(),
          new THREE.Vector3(),
        ]);

    const stringLine =
      new THREE.Line(
        stringGeom,
        stringMat
      );

    scene.add(
      stringLine
    );

    const challenge =
      randomChallenge();

    stateRef.current.pos.copy(
      challenge.pos
    );

    stateRef.current.quat.copy(
      challenge.quat
    );

    movingFrame.position.copy(
      challenge.pos
    );

    movingFrame.quaternion.copy(
      challenge.quat
    );

    updateReadouts();

    function updateCamera() {
      const {
        theta,
        phi,
        radius,
      } = sphericalRef.current;

      camera.position.set(
        radius *
          Math.sin(phi) *
          Math.sin(theta),

        radius *
          Math.cos(phi),

        radius *
          Math.sin(phi) *
          Math.cos(theta)
      );

      camera.lookAt(
        0,
        -0.3,
        0
      );
    }

    updateCamera();

    let rafId;

    function animate() {
      rafId =
        requestAnimationFrame(
          animate
        );

      movingShadow.position.set(
        movingFrame.position.x,
        FLOOR_Y + 0.003,
        movingFrame.position.z
      );

      stringLine.geometry.setFromPoints([
        new THREE.Vector3(
          0,
          0,
          0
        ),

        movingFrame.position,
      ]);

      stringLine.computeLineDistances();

      renderer.render(
        scene,
        camera
      );
    }

    animate();

    function onPointerDown(e) {
      dragRef.current = {
        dragging: true,
        x: e.clientX,
        y: e.clientY,
      };

      dom.style.cursor =
        "grabbing";
    }

    function onPointerMove(e) {
      if (
        !dragRef.current.dragging
      )
        return;

      const dx =
        e.clientX -
        dragRef.current.x;

      const dy =
        e.clientY -
        dragRef.current.y;

      dragRef.current.x =
        e.clientX;

      dragRef.current.y =
        e.clientY;

      sphericalRef.current.theta -=
        dx * 0.006;

      sphericalRef.current.phi =
        THREE.MathUtils.clamp(
          sphericalRef.current.phi -
            dy * 0.006,
          0.25,
          Math.PI - 0.3
        );

      updateCamera();
    }

    function onPointerUp() {
      dragRef.current.dragging =
        false;

      if (dom) {
        dom.style.cursor =
          "grab";
      }
    }

    function onWheel(e) {
      e.preventDefault();

      sphericalRef.current.radius =
        THREE.MathUtils.clamp(
          sphericalRef.current.radius +
            e.deltaY * 0.012,
          7,
          24
        );

      updateCamera();
    }

    function onResize() {
      if (!mountRef.current)
        return;

      const w =
        mountRef.current.clientWidth ||
        320;

      const h =
        mountRef.current.clientHeight ||
        320;

      camera.aspect =
        w / h;

      camera.updateProjectionMatrix();

      renderer.setSize(
        w,
        h,
        false
      );
    }

    const dom =
      renderer.domElement;

    dom.style.touchAction =
      "none";

    dom.style.cursor =
      "grab";

    dom.addEventListener(
      "pointerdown",
      onPointerDown
    );

    window.addEventListener(
      "pointermove",
      onPointerMove
    );

    window.addEventListener(
      "pointerup",
      onPointerUp
    );

    dom.addEventListener(
      "wheel",
      onWheel,
      { passive: false }
    );

    window.addEventListener(
      "resize",
      onResize
    );

    const resizeObserver =
      new ResizeObserver(
        onResize
      );

    resizeObserver.observe(
      mount
    );

    return () => {
      cancelAnimationFrame(
        rafId
      );

      if (
        rotationAnimationRef.current
      ) {
        cancelAnimationFrame(
          rotationAnimationRef.current
        );

        rotationAnimationRef.current =
          null;
      }

      dom.removeEventListener(
        "pointerdown",
        onPointerDown
      );

      window.removeEventListener(
        "pointermove",
        onPointerMove
      );

      window.removeEventListener(
        "pointerup",
        onPointerUp
      );

      dom.removeEventListener(
        "wheel",
        onWheel
      );

      window.removeEventListener(
        "resize",
        onResize
      );

      resizeObserver.disconnect();

      if (mount.contains(dom)) {
        mount.removeChild(dom);
      }

      renderer.dispose();
    };
  }, [updateReadouts]);

  const translate = (
    axis,
    sign
  ) => {
    const delta =
      new THREE.Vector3();

    delta[axis] =
      sign * TRANS_STEP;

    stateRef.current.pos.add(
      delta
    );

    groupRef.current.position.copy(
      stateRef.current.pos
    );

    setMoves(
      (c) => c + 1
    );

    updateReadouts();
  };

  // ============================================================
  // ANIMATED 90° ROTATION
  // ============================================================

  const rotate = (
    axis,
    sign
  ) => {
    const vec =
      axis === "x"
        ? new THREE.Vector3(
            1,
            0,
            0
          )
        : axis === "y"
        ? new THREE.Vector3(
            0,
            1,
            0
          )
        : new THREE.Vector3(
            0,
            0,
            1
          );

    // Exactly 90°.
    const deltaQ =
      new THREE.Quaternion()
        .setFromAxisAngle(
          vec,
          sign * ROT_STEP
        );

    // IMPORTANT:
    // multiply() means rotation about
    // the frame's LOCAL axis.
    const targetQuat =
      stateRef.current.quat
        .clone()
        .multiply(deltaQ);

    // Cancel a previous animation
    // if the player clicks again quickly.
    if (
      rotationAnimationRef.current
    ) {
      cancelAnimationFrame(
        rotationAnimationRef.current
      );

      rotationAnimationRef.current =
        null;
    }

    // Start from the orientation
    // that is currently visible.
    const startQuat =
      groupRef.current.quaternion.clone();

    const startTime =
      performance.now();

    const animateRotation = (
      now
    ) => {
      const elapsed =
        now - startTime;

      const progress =
        THREE.MathUtils.clamp(
          elapsed /
            ROT_ANIMATION_MS,
          0,
          1
        );

      // Smooth ease-in-out.
      const eased =
        progress < 0.5
          ? 2 *
            progress *
            progress
          : 1 -
            Math.pow(
              -2 * progress + 2,
              2
            ) /
              2;

      // Smooth quaternion interpolation.
      groupRef.current.quaternion.slerpQuaternions(
        startQuat,
        targetQuat,
        eased
      );

      // Keep state synchronized
      // with the animated orientation.
      stateRef.current.quat.copy(
        groupRef.current.quaternion
      );

      updateReadouts();

      if (
        progress < 1
      ) {
        rotationAnimationRef.current =
          requestAnimationFrame(
            animateRotation
          );
      } else {
        // Force the exact final orientation.
        stateRef.current.quat.copy(
          targetQuat
        );

        groupRef.current.quaternion.copy(
          targetQuat
        );

        rotationAnimationRef.current =
          null;

        updateReadouts();
      }
    };

    rotationAnimationRef.current =
      requestAnimationFrame(
        animateRotation
      );

    setMoves(
      (c) => c + 1
    );
  };

  const reset = () => {
    // Stop any running animation.
    if (
      rotationAnimationRef.current
    ) {
      cancelAnimationFrame(
        rotationAnimationRef.current
      );

      rotationAnimationRef.current =
        null;
    }

    const challenge =
      randomChallenge();

    stateRef.current.pos.copy(
      challenge.pos
    );

    stateRef.current.quat.copy(
      challenge.quat
    );

    groupRef.current.position.copy(
      challenge.pos
    );

    groupRef.current.quaternion.copy(
      challenge.quat
    );

    setMoves(0);

    updateReadouts();
  };

  const axes = [
    "x",
    "y",
    "z",
  ];

  const cardStyle = {
    borderColor: INK,
    background: "#FFFDF7",
    boxShadow:
      `5px 5px 0px ${INK}`,
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@500;600;700;800&family=Nunito:wght@500;700;800&display=swap');

        * {
          box-sizing: border-box;
        }

        html,
        body,
        #root {
          margin: 0;
          min-height: 100%;
          width: 100%;
        }

        body {
          overflow-x: hidden;
        }

        .app {
          min-height: 100dvh;
          width: 100%;
          padding: 24px;
          display: flex;
          justify-content: center;
          align-items: center;
          background:
            linear-gradient(
              160deg,
              #FFE8D6 0%,
              #FFF4E0 45%,
              #E9F7EF 100%
            );

          font-family:
            'Nunito',
            ui-sans-serif,
            system-ui,
            sans-serif;
        }

        .app-shell {
          width: 100%;
          max-width: 1200px;

          display: grid;

          grid-template-columns:
            minmax(0, 1.35fr)
            minmax(340px, 0.85fr);

          gap: 24px;
          align-items: start;
        }

        .scene-card,
        .control-card {
          min-width: 0;
        }

        .scene-card {
          border: 4px solid ${INK};
          border-radius: 28px;
          background: #FFFDF7;
          box-shadow: 6px 6px 0px ${INK};
          overflow: hidden;
        }

        .scene-header {
          min-height: 68px;
          padding: 16px 20px 10px;

          display: flex;
          align-items: center;
          justify-content: space-between;

          gap: 12px;
        }

        .scene-title {
          color: ${INK};

          font-family:
            'Baloo 2',
            'Nunito',
            sans-serif;

          font-size: 21px;
          line-height: 1.1;
          font-weight: 800;
        }

        .status-pill {
          flex-shrink: 0;

          display: flex;
          align-items: center;

          gap: 6px;

          padding:
            6px 10px;

          border:
            2px solid ${INK};

          border-radius: 999px;

          color: ${INK};

          font-family:
            'Baloo 2',
            'Nunito',
            sans-serif;

          font-size: 12px;
          font-weight: 800;

          text-transform: uppercase;
          white-space: nowrap;
        }

        .scene-container {
          width: 100%;
          height:
            clamp(
              300px,
              42vw,
              480px
            );

          min-height: 280px;
        }

        .scene-footer {
          min-height: 48px;

          padding:
            8px 20px 14px;

          display: flex;
          align-items: center;
          justify-content: space-between;

          gap: 12px;

          color: ${INK};
        }

        .legend {
          display: flex;
          align-items: center;

          gap: 16px;

          font-size: 12px;
          white-space: nowrap;
        }

        .legend-item {
          display: flex;
          align-items: center;
          gap: 5px;
        }

        .legend-dot {
          width: 11px;
          height: 11px;

          border:
            2px solid ${INK};

          border-radius: 50%;
          flex-shrink: 0;
        }

        .scene-hint {
          font-size: 11px;
          opacity: 0.55;
          text-align: right;
        }

        .control-card {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }

        .panel {
          border:
            4px solid ${INK};

          border-radius: 24px;

          background: #FFFDF7;

          box-shadow:
            5px 5px 0px ${INK};
        }

        .status-panel {
          padding: 16px 18px;

          display: grid;

          grid-template-columns:
            repeat(3, 1fr);

          gap: 12px;
        }

        .status-item {
          min-width: 0;
        }

        .status-label {
          margin-bottom: 2px;

          color: ${INK};

          opacity: 0.6;

          font-family:
            'Baloo 2',
            'Nunito',
            sans-serif;

          font-size: 11px;
          font-weight: 700;

          text-transform: uppercase;
          letter-spacing: 0.04em;
        }

        .status-value {
          color: ${INK};

          font-size: 21px;
          line-height: 1.1;
          font-weight: 800;

          font-variant-numeric:
            tabular-nums;
        }

        .controls-panel {
          padding:
            16px 18px;
        }

        .panel-title {
          margin-bottom: 12px;

          color: ${INK};

          font-family:
            'Baloo 2',
            'Nunito',
            sans-serif;

          font-size: 16px;
          font-weight: 800;
        }

        .axis-list {
          display: flex;
          flex-direction: column;
          gap: 10px;
        }

        .axis-row {
          min-width: 0;

          display: grid;

          grid-template-columns:
            32px
            minmax(0, 1fr)
            minmax(0, 1fr);

          gap: 10px;

          align-items: center;

          padding: 9px;

          border:
            2px solid ${INK};

          border-radius: 17px;
        }

        .axis-badge {
          width: 30px;
          height: 30px;

          display: flex;
          align-items: center;
          justify-content: center;

          border:
            2px solid ${INK};

          border-radius: 50%;

          color: white;

          font-family:
            'Baloo 2',
            'Nunito',
            sans-serif;

          font-size: 15px;
          font-weight: 800;
        }

        .button-group {
          min-width: 0;

          display: grid;

          grid-template-columns:
            1fr 1fr;

          gap: 7px;
        }

        .group-label {
          grid-column: 1 / -1;

          margin-bottom: -2px;

          color: ${INK};

          opacity: 0.5;

          font-size: 9px;
          font-weight: 800;

          text-align: center;

          text-transform: uppercase;
          letter-spacing: 0.05em;
        }

        .toy-btn {
          min-width: 0;
          height: 38px;

          display: flex;
          align-items: center;
          justify-content: center;

          border:
            2px solid ${INK};

          border-radius: 11px;

          background: white;
          color: ${INK};

          box-shadow:
            2px 2px 0px ${INK};

          cursor: pointer;

          transition:
            transform 0.08s ease,
            box-shadow 0.08s ease,
            background 0.15s ease;

          touch-action: manipulation;

          -webkit-tap-highlight-color:
            transparent;
        }

        .toy-btn:hover {
          background: #FFF9E8;
        }

        .toy-btn:active {
          transform:
            translate(
              2px,
              2px
            );

          box-shadow:
            0px 0px 0px ${INK};
        }

        .controls-note {
          display: block;

          margin-top: 12px;

          color: ${INK};

          opacity: 0.5;

          font-size: 10px;
          line-height: 1.4;
        }

        .matrix-panel {
          padding:
            16px 18px;
        }

        .matrix-wrapper {
          display: flex;

          align-items: center;
          justify-content: center;

          gap: 8px;

          margin-bottom: 16px;
        }

        .bracket {
          color: ${INK};
          opacity: 0.3;

          font-size: 38px;
          font-weight: 200;
          line-height: 1;
        }

        .matrix-grid {
          display: grid;

          grid-template-columns:
            repeat(
              3,
              minmax(
                45px,
                1fr
              )
            );

          gap:
            6px 12px;
        }

        .matrix-value {
          width: 100%;

          color: ${INK};

          font-size: 13px;
          font-weight: 800;

          text-align: right;

          font-variant-numeric:
            tabular-nums;
        }

        .translation-title {
          margin-bottom: 8px;

          color: ${INK};

          font-family:
            'Baloo 2',
            'Nunito',
            sans-serif;

          font-size: 14px;
          font-weight: 800;
        }

        .translation-values {
          display: grid;

          grid-template-columns:
            repeat(3, 1fr);

          gap: 8px;
        }

        .translation-value {
          color: ${INK};

          font-size: 13px;
          font-weight: 800;

          font-variant-numeric:
            tabular-nums;
        }

        .translation-axis {
          font-weight: 800;
        }

        .reset-btn {
          min-height: 50px;
          width: 100%;

          display: flex;
          align-items: center;
          justify-content: center;

          gap: 8px;

          border:
            4px solid ${INK};

          border-radius: 18px;

          background: #FFD43B;
          color: ${INK};

          box-shadow:
            5px 5px 0px ${INK};

          font-family:
            'Baloo 2',
            'Nunito',
            sans-serif;

          font-size: 14px;
          font-weight: 800;

          text-transform: uppercase;
          letter-spacing: 0.04em;

          cursor: pointer;

          transition:
            transform 0.08s ease,
            box-shadow 0.08s ease;

          touch-action: manipulation;
        }

        .reset-btn:hover {
          background: #FFDC5C;
        }

        .reset-btn:active {
          transform:
            translate(
              3px,
              3px
            );

          box-shadow:
            1px 1px 0px ${INK};
        }

        @keyframes pop {
          0% {
            transform:
              scale(0.6)
              rotate(-6deg);

            opacity: 0;
          }

          55% {
            transform:
              scale(1.12)
              rotate(3deg);

            opacity: 1;
          }

          100% {
            transform:
              scale(1)
              rotate(0deg);

            opacity: 1;
          }
        }

        .pop-anim {
          animation:
            pop 0.5s
            cubic-bezier(
              .34,
              1.56,
              .64,
              1
            );
        }

        @media (max-width: 1000px) {
          .app {
            padding: 18px;
          }

          .app-shell {
            grid-template-columns:
              minmax(0, 1.1fr)
              minmax(300px, 0.9fr);

            gap: 18px;
          }

          .scene-container {
            height: 400px;
          }

          .axis-row {
            gap: 7px;
          }

          .toy-btn {
            height: 36px;
          }
        }

        @media (max-width: 780px) {
          .app {
            width: 100%;
            height: 100dvh;
            min-height: 100dvh;

            padding: 8px;

            overflow: hidden;

            align-items: stretch;
          }

          .app-shell {
            width: 100%;
            height: 100%;
            max-width: 620px;

            display: grid;

            grid-template-columns: 1fr;

            grid-template-rows:
              45% 55%;

            gap: 8px;

            overflow: hidden;
          }

          .scene-card {
            min-height: 0;
            height: 100%;

            border-width: 3px;
            border-radius: 18px;

            box-shadow:
              4px 4px 0px ${INK};

            display: flex;
            flex-direction: column;

            overflow: hidden;
          }

          .scene-header {
            flex: 0 0 auto;

            min-height: 42px;
            height: 42px;

            padding:
              7px 10px 4px;

            gap: 6px;
          }

          .scene-title {
            font-size: 15px;
            line-height: 1;
          }

          .status-pill {
            padding:
              3px 6px;

            border-width: 2px;

            border-radius: 999px;

            font-size: 8px;

            gap: 3px;
          }

          .status-pill svg {
            width: 10px;
            height: 10px;
          }

          .scene-container {
            flex:
              1 1 auto;

            width: 100%;

            min-height: 0;

            height: auto;
          }

          .scene-footer {
            flex:
              0 0 auto;

            min-height: 28px;
            height: 28px;

            padding:
              3px 9px 5px;

            gap: 5px;
          }

          .legend {
            gap: 7px;
            font-size: 8px;
          }

          .legend-dot {
            width: 7px;
            height: 7px;

            border-width: 1px;
          }

          .scene-hint {
            display: none;
          }

          .control-card {
            min-height: 0;
            height: 100%;

            display: flex;
            flex-direction: column;

            gap: 7px;

            overflow-y: auto;
            overflow-x: hidden;

            padding-right: 2px;

            scrollbar-width: thin;
          }

          .panel {
            flex:
              0 0 auto;

            border-width: 3px;

            border-radius: 16px;

            box-shadow:
              3px 3px 0px ${INK};
          }

          .status-panel {
            padding:
              7px 9px;

            min-height: 45px;

            grid-template-columns:
              repeat(3, 1fr);

            gap: 6px;
          }

          .status-label {
            margin-bottom: 0;

            font-size: 7px;
            line-height: 1;
          }

          .status-value {
            font-size: 14px;
            line-height: 1.15;
          }

          .controls-panel {
            padding:
              8px 9px;
          }

          .panel-title {
            margin-bottom: 6px;

            font-size: 12px;
            line-height: 1;
          }

          .axis-list {
            gap: 5px;
          }

          .axis-row {
            grid-template-columns:
              24px
              minmax(0, 1fr)
              minmax(0, 1fr);

            gap: 5px;

            padding:
              4px 5px;

            border-width: 2px;
            border-radius: 10px;
          }

          .axis-badge {
            width: 23px;
            height: 23px;

            border-width: 2px;

            font-size: 11px;
          }

          .button-group {
            gap: 3px;
          }

          .group-label {
            font-size: 6px;
            line-height: 1;
          }

          .toy-btn {
            width: 100%;
            height: 27px;

            border-width: 2px;
            border-radius: 7px;

            box-shadow:
              1px 1px 0px ${INK};
          }

          .toy-btn svg {
            width: 12px;
            height: 12px;
          }

          .toy-btn:active {
            transform:
              translate(
                1px,
                1px
              );

            box-shadow:
              0px 0px 0px ${INK};
          }

          .controls-note {
            margin-top: 5px;

            font-size: 7px;
            line-height: 1.2;
          }

          .matrix-panel {
            padding:
              8px 9px;

            min-height: 115px;
          }

          .matrix-wrapper {
            width: 100%;

            justify-content: center;

            gap: 4px;

            margin-bottom: 8px;

            overflow-x: auto;
            overflow-y: hidden;

            padding-bottom: 2px;
          }

          .bracket {
            flex-shrink: 0;

            font-size: 27px;
          }

          .matrix-grid {
            flex-shrink: 0;

            grid-template-columns:
              repeat(
                3,
                42px
              );

            gap:
              3px 6px;
          }

          .matrix-value {
            width: 42px;

            font-size: 9px;
          }

          .translation-title {
            margin-bottom: 4px;

            font-size: 10px;
          }

          .translation-values {
            gap: 5px;
          }

          .translation-value {
            font-size: 9px;
          }

          .reset-btn {
            flex:
              0 0 auto;

            min-height: 35px;
            height: 35px;

            border-width: 3px;
            border-radius: 12px;

            box-shadow:
              3px 3px 0px ${INK};

            font-size: 9px;

            gap: 5px;
          }

          .reset-btn svg {
            width: 13px;
            height: 13px;
          }
        }

        @media (max-width: 480px) {
          .app {
            padding: 5px;
          }

          .app-shell {
            grid-template-rows:
              44% 56%;

            gap: 6px;
          }

          .scene-card {
            border-radius: 15px;

            box-shadow:
              3px 3px 0px ${INK};
          }

          .scene-header {
            min-height: 36px;
            height: 36px;

            padding:
              5px 8px 3px;
          }

          .scene-title {
            font-size: 13px;
          }

          .status-pill {
            padding:
              2px 5px;

            font-size: 7px;
          }

          .scene-footer {
            min-height: 23px;
            height: 23px;

            padding:
              2px 7px 3px;
          }

          .legend {
            font-size: 7px;
            gap: 5px;
          }

          .legend-dot {
            width: 6px;
            height: 6px;
          }

          .status-panel {
            min-height: 39px;

            padding:
              6px 7px;

            border-radius: 12px;
          }

          .status-label {
            font-size: 6px;
          }

          .status-value {
            font-size: 12px;
          }

          .controls-panel {
            padding: 7px;

            border-radius: 12px;
          }

          .panel-title {
            font-size: 10px;

            margin-bottom: 5px;
          }

          .axis-list {
            gap: 4px;
          }

          .axis-row {
            grid-template-columns:
              21px
              minmax(0, 1fr)
              minmax(0, 1fr);

            gap: 4px;

            padding:
              3px 4px;

            border-radius: 8px;
          }

          .axis-badge {
            width: 20px;
            height: 20px;

            font-size: 9px;
          }

          .group-label {
            font-size: 5px;
          }

          .toy-btn {
            height: 24px;
            border-radius: 6px;
          }

          .toy-btn svg {
            width: 10px;
            height: 10px;
          }

          .controls-note {
            font-size: 6px;
            margin-top: 4px;
          }

          .matrix-panel {
            min-height: 105px;

            padding: 7px;

            border-radius: 12px;
          }

          .matrix-wrapper {
            margin-bottom: 6px;
          }

          .bracket {
            font-size: 24px;
          }

          .matrix-grid {
            grid-template-columns:
              repeat(
                3,
                38px
              );

            gap:
              3px 5px;
          }

          .matrix-value {
            width: 38px;
            font-size: 8px;
          }

          .translation-title {
            font-size: 9px;
            margin-bottom: 3px;
          }

          .translation-value {
            font-size: 8px;
          }

          .reset-btn {
            height: 31px;
            min-height: 31px;

            border-radius: 10px;

            font-size: 8px;

            box-shadow:
              2px 2px 0px ${INK};
          }

          .reset-btn svg {
            width: 11px;
            height: 11px;
          }
        }

        @media (max-width: 350px) {
          .app-shell {
            grid-template-rows:
              42% 58%;
          }

          .scene-title {
            font-size: 11px;
          }

          .status-pill {
            font-size: 6px;
          }

          .axis-row {
            grid-template-columns:
              19px
              minmax(0, 1fr)
              minmax(0, 1fr);
          }

          .axis-badge {
            width: 18px;
            height: 18px;

            font-size: 8px;
          }

          .toy-btn {
            height: 22px;
          }

          .toy-btn svg {
            width: 9px;
            height: 9px;
          }

          .status-value {
            font-size: 11px;
          }

          .matrix-grid {
            grid-template-columns:
              repeat(
                3,
                34px
              );
          }

          .matrix-value {
            width: 34px;
            font-size: 7px;
          }
        }
      `}</style>

      <main className="app">
        <div className="app-shell">

          {/* ================= SCENE ================= */}

          <section className="scene-card">

            <header className="scene-header">

              <div className="scene-title">
                Match the Wobbly Frame!
              </div>

              <div
                className={`status-pill ${
                  pop
                    ? "pop-anim"
                    : ""
                }`}
                style={{
                  background:
                    aligned
                      ? "#B2F2BB"
                      : "#FFF3BF",
                }}
              >
                {aligned ? (
                  <PartyPopper
                    size={13}
                  />
                ) : (
                  <Sparkles
                    size={13}
                  />
                )}

                <span>
                  {aligned
                    ? "Matched!"
                    : "Keep going"}
                </span>
              </div>

            </header>

            <div
              ref={mountRef}
              className="scene-container"
            />

            <footer className="scene-footer">

              <div className="legend">

                <div className="legend-item">

                  <span
                    className="legend-dot"
                    style={{
                      background:
                        "#FFD43B",
                    }}
                  />

                  <span>
                    target
                  </span>

                </div>

                <div className="legend-item">

                  <span
                    className="legend-dot"
                    style={{
                      background:
                        "#63E6E8",
                    }}
                  />

                  <span>
                    your frame
                  </span>

                </div>

              </div>

              <div className="scene-hint">
                drag to spin ·
                scroll to zoom
              </div>

            </footer>

          </section>

          {/* ================= CONTROLS ================= */}

          <section className="control-card">

            {/* STATUS */}

            <div
              className="panel status-panel"
              style={cardStyle}
            >

              <div className="status-item">

                <div className="status-label">
                  Distance
                </div>

                <div className="status-value">
                  {fmt(posErr)}
                </div>

              </div>

              <div className="status-item">

                <div className="status-label">
                  Tilt
                </div>

                <div className="status-value">
                  {fmt(angErr)}°
                </div>

              </div>

              <div className="status-item">

                <div className="status-label">
                  Moves
                </div>

                <div className="status-value">
                  {moves}
                </div>

              </div>

            </div>

            {/* AXIS CONTROLS */}

            <div
              className="panel controls-panel"
              style={cardStyle}
            >

              <div className="panel-title">
                Push &amp; Spin
              </div>

              <div className="axis-list">

                {axes.map(
                  (ax) => (
                    <div
                      key={ax}
                      className="axis-row"
                      style={{
                        background:
                          `${AXIS[ax].hex}22`,
                      }}
                    >

                      <div
                        className="axis-badge"
                        style={{
                          background:
                            AXIS[ax].hex,
                        }}
                      >
                        {AXIS[ax].label}
                      </div>

                      {/* TRANSLATION */}

                      <div className="button-group">

                        <div className="group-label">
                          Slide
                        </div>

                        <button
                          className="toy-btn"
                          onClick={() =>
                            translate(
                              ax,
                              -1
                            )
                          }
                          title={`Slide -${ax.toUpperCase()}`}
                          aria-label={`Slide negative ${ax} axis`}
                        >
                          <Minus
                            size={15}
                            strokeWidth={3}
                          />
                        </button>

                        <button
                          className="toy-btn"
                          onClick={() =>
                            translate(
                              ax,
                              1
                            )
                          }
                          title={`Slide +${ax.toUpperCase()}`}
                          aria-label={`Slide positive ${ax} axis`}
                        >
                          <Plus
                            size={15}
                            strokeWidth={3}
                          />
                        </button>

                      </div>

                      {/* ROTATION */}

                      <div className="button-group">

                        <div className="group-label">
                          Spin
                        </div>

                        <button
                          className="toy-btn"
                          onClick={() =>
                            rotate(
                              ax,
                              -1
                            )
                          }
                          title={`Spin -90° about ${ax.toUpperCase()}`}
                          aria-label={`Rotate negative ${ax} axis`}
                        >
                          <RotateCcw
                            size={15}
                            strokeWidth={3}
                          />
                        </button>

                        <button
                          className="toy-btn"
                          onClick={() =>
                            rotate(
                              ax,
                              1
                            )
                          }
                          title={`Spin +90° about ${ax.toUpperCase()}`}
                          aria-label={`Rotate positive ${ax} axis`}
                        >
                          <RotateCw
                            size={15}
                            strokeWidth={3}
                          />
                        </button>

                      </div>

                    </div>
                  )
                )}

              </div>

              <span className="controls-note">
                Each push moves{" "}
                {TRANS_STEP} units.
                Each spin turns 90°
                around the frame's own
                axes with smooth animation.
              </span>

            </div>

            {/* ROTATION MATRIX */}

            <div
              className="panel matrix-panel"
              style={cardStyle}
            >

              <div className="panel-title">
                Rotation Matrix R
              </div>

              <div className="matrix-wrapper">

                <span className="bracket">
                  [
                </span>

                <div className="matrix-grid">

                  {matrixEls.map(
                    (v, i) => (
                      <span
                        key={i}
                        className="matrix-value"
                      >
                        {fmt(v)}
                      </span>
                    )
                  )}

                </div>

                <span className="bracket">
                  ]
                </span>

              </div>

              <div className="translation-title">
                Translation t
              </div>

              <div className="translation-values">

                {["x", "y", "z"].map(
                  (ax, i) => (
                    <div
                      key={ax}
                      className="translation-value"
                    >

                      <span
                        className="translation-axis"
                        style={{
                          color:
                            AXIS[ax].dark,
                        }}
                      >
                        {ax}
                      </span>{" "}

                      {fmt(
                        translation[i]
                      )}

                    </div>
                  )
                )}

              </div>

            </div>

            {/* RESET */}

            <button
              className="reset-btn"
              onClick={reset}
            >

              <Shuffle
                size={17}
                strokeWidth={3}
              />

              New Wobbly Frame

            </button>

          </section>

        </div>
      </main>
    </>
  );
}