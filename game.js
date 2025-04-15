// Declare images for defenders, attackers, and shot indicators
let defenderImg = new Image();
let attackerImg = new Image();
let redSquareImg = new Image();
let blueSquareImg = new Image();

// Server URL for logging/predictions and connection flag
const LOGGER_SERVER = 'http://localhost:5001';
let isLoggerConnected = false;

// Constants for game grid and cell sizing
const GRID_SIZE = 10;
const CELL_SIZE = 80;

// Prediction colors for each attacker (A, B, and C) with primary and secondary opacities
const PREDICTION_COLORS = {
  A: {
    primary: 'rgba(139, 0, 0, 0.5)',    // Dark red for primary prediction
    secondary: 'rgba(255, 102, 102, 0.3)'  // Light red for secondary prediction
  },
  B: {
    primary: 'rgba(34, 89, 34, 0.5)',     // Forest green for primary prediction
    secondary: 'rgba(135, 162, 127, 0.3)'  // Sage green for secondary prediction
  },
  C: {
    primary: 'rgba(0, 0, 139, 0.5)',     // Dark blue for primary prediction
    secondary: 'rgba(100, 149, 237, 0.3)'  // Light blue for secondary prediction
  }
};

// Global game state variables
let board = [];
let attackers = [];
let trainingData = [];
let hoveredCell = null;
let gameOver = false;
let actions = [];
let showPaths = false;
let autoPlayActive = false;
let autoPlayInterval = null;
const TURN_DELAY_MS = 100;
let attackerHistory = {};

// Maintain shot history for defenders A and B
let defenderShotHistory = {
  A: [
    [-1, -1],
    [-1, -1],
    [-1, -1],
    [-1, -1],
  ],
  B: [
    [-1, -1],
    [-1, -1],
    [-1, -1],
    [-1, -1],
  ],
};

// Function to load game images and ensure they're available before starting the game
function loadGameImages() {
    console.log("Loading game images...");
    
    // Create promises for image loading
    const defenderPromise = new Promise((resolve, reject) => {
        defenderImg.onload = () => {
            console.log("Defender image loaded successfully");
            resolve();
        };
        defenderImg.onerror = () => {
            console.error("Error loading Defender image");
            reject();
        };
    });

    const attackerPromise = new Promise((resolve, reject) => {
        attackerImg.onload = () => {
            console.log("Attacker image loaded successfully");
            resolve();
        };
        attackerImg.onerror = () => {
            console.error("Error loading Attacker image");
            reject();
        };
    });

    // Load the red square image (for shot visualization) with logging
    redSquareImg.onload = function() {
        console.log("Red square image loaded successfully");
    };
    
    redSquareImg.onerror = function() {
        console.error("Error loading red square image");
    };

    // Load the blue square image with logging
    blueSquareImg.onload = function() {
        console.log("Blue square image loaded successfully");
    };
    
    blueSquareImg.onerror = function() {
        console.error("Error loading blue square image");
    };

    // Set source paths for defender and attacker images
    defenderImg.src = "Defender.png";
    attackerImg.src = "Attacker.png";

    // Return promise that resolves when both defender and attacker images are loaded
    return Promise.all([defenderPromise, attackerPromise]);
}

// Start the game once the window has loaded images
window.onload = function() {
    loadGameImages().then(() => {
        console.log("All images loaded, starting game");
        newGame();
    }).catch(error => {
        console.error("Error loading images:", error);
        newGame(); 
    });
};

// Get references to canvas and game control elements in the HTML document
const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");
const newGameBtn = document.getElementById("newGameBtn");
const nextTurnBtn = document.getElementById("nextTurnBtn");
const actionLogBtn = document.getElementById("actionLogBtn");
const togglePathsBtn = document.getElementById("togglePathsBtn");
const statusMessage = document.getElementById("statusMessage");
const actionLog = document.getElementById("actionLog");
const autoPlayBtn = document.getElementById("autoPlayBtn");
const generatePredictionsBtn = document.getElementById("generatePredictionsBtn");
const togglePredictionOutputBtn = document.getElementById("togglePredictionOutputBtn");
const predictionOutput = document.getElementById("prediction-output");

// Toggle Auto Play functionality: Start or stop automatic turns
function toggleAutoPlay() {
  if (autoPlayInterval) {
    stopAutoPlay();
  } else {
    startAutoPlay();
  }
}

// Start automatic turns (auto-play mode)
function startAutoPlay() {
  autoPlayActive = true;
  autoPlayBtn.textContent = 'Stop Auto Play';
  autoPlayBtn.style.backgroundColor = '#f44336'; 
  
  newGameBtn.disabled = true;
  nextTurnBtn.disabled = true;
  
  autoPlayInterval = setInterval(() => {
    if (gameOver) {
      newGame(); // Automatically start a new game if the current one is over
    } else {
      nextTurn(); // Process the next turn
    }
  }, TURN_DELAY_MS);
}

// Stop automatic turns (auto-play mode)
function stopAutoPlay() {
  autoPlayActive = false;
  autoPlayBtn.textContent = 'Auto Play';
  autoPlayBtn.style.backgroundColor = '#333';
  
  newGameBtn.disabled = false;
  nextTurnBtn.disabled = false;
  
  if (autoPlayInterval) {
    clearInterval(autoPlayInterval);
    autoPlayInterval = null;
  }
}

// Create and return an empty board (2D array) with GRID_SIZE rows and columns
function createEmptyBoard() {
  let arr = [];
  for (let r = 0; r < GRID_SIZE; r++) {
    arr[r] = [];
    for (let c = 0; c < GRID_SIZE; c++) {
      arr[r][c] = 0;
    }
  }
  return arr;
}

// Place defenders on the board randomly on the first row
function placeDefenders(boardArr) {
  let pos1 = Math.floor(Math.random() * 10);
  let pos2 = Math.floor(Math.random() * 10);
  
  while (pos1 === pos2) {
    pos2 = Math.floor(Math.random() * 10);
  }
  
   // Assign defender labels based on positions
  const Apos = Math.min(pos1, pos2);
  const Bpos = Math.max(pos1, pos2);
  
  boardArr[0][Apos] = "A";
  boardArr[0][Bpos] = "B";
}

// Generate a straight path using Bresenham's algorithm from one point to another
function straightPath(r0, c0, r1, c1) {
  let path = [[r0, c0]];
  let currentR = r0;
  let currentC = c0;
  
  // Calculate direction and absolute differences
  const dr = r1 - r0;
  const dc = c1 - c0;
  const stepR = dr > 0 ? 1 : -1;
  const stepC = dc > 0 ? 1 : -1;
  const absDr = Math.abs(dr);
  const absDc = Math.abs(dc);
  
  // Bresenham's algorithm implementation
  let error = 0;
  let verticalFirst = absDr > absDc; // Prioritize major axis
  
  while (currentR !== r1 || currentC !== c1) {
      if (verticalFirst) {
          if (currentR !== r1) {
              currentR += stepR;
              path.push([currentR, currentC]);
              error += absDc;
          }
          if (error >= absDr && currentC !== c1) {
              currentC += stepC;
              path.push([currentR, currentC]);
              error -= absDr;
          }
      } else {
          if (currentC !== c1) {
              currentC += stepC;
              path.push([currentR, currentC]);
              error += absDr;
          }
          if (error >= absDc && currentR !== r1) {
              currentR += stepR;
              path.push([currentR, currentC]);
              error -= absDc;
          }
      }
  }
  
  return path;
}

// Generate a random (curve) path between two points by shuffling vertical and horizontal moves
function RandomPath(r0, c0, r1, c1) {
  let dr = r1 - r0;
  let dc = c1 - c0;

  let verticalSteps = [];
  let horizontalSteps = [];

  // Determine vertical steps direction
  if (dr > 0) {
      verticalSteps = Array(dr).fill('down');
  } else if (dr < 0) {
      verticalSteps = Array(-dr).fill('up');
  }

  // Determine horizontal steps direction
  if (dc > 0) {
      horizontalSteps = Array(dc).fill('right');
  } else if (dc < 0) {
      horizontalSteps = Array(-dc).fill('left');
  }

  // Combine and shuffle the steps
  let allSteps = verticalSteps.concat(horizontalSteps);
  for (let i = allSteps.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [allSteps[i], allSteps[j]] = [allSteps[j], allSteps[i]];
  }

  // Generate the path
  let currentR = r0;
  let currentC = c0;
  let path = [[currentR, currentC]];

  for (const step of allSteps) {
      switch (step) {
          case 'up':
              currentR -= 1;
              break;
          case 'down':
              currentR += 1;
              break;
          case 'left':
              currentC -= 1;
              break;
          case 'right':
              currentC += 1;
              break;
      }
      path.push([currentR, currentC]);
  }

  // Ensure the path ends at the target
  const lastStep = path[path.length - 1];
  if (lastStep[0] !== r1 || lastStep[1] !== c1) {
      path.push([r1, c1]);
  }

  return path;
}

// Finds the nearest defender to a given spawn position using Manhattan distance
function nearestDefender(spawn) {
  let def1 = [8, 2, "A"],
      def2 = [7, 7, "B"];
  let dist1 = Math.abs(def1[0] - spawn[0]) + Math.abs(def1[1] - spawn[1]);
  let dist2 = Math.abs(def2[0] - spawn[0]) + Math.abs(def2[1] - spawn[1]);
  return dist1 <= dist2 ? def1 : def2;
}

// Place attackers on the board and assign them random paths towards defenders
function placeAttackers() {
  attackers = [];
  let usedCols = [];
   // Randomly pick three unique columns for attacker spawn positions
  while (usedCols.length < 3) {
    let randCol = Math.floor(Math.random() * GRID_SIZE);
    if (!usedCols.includes(randCol)) usedCols.push(randCol);
  }
  usedCols.sort((a, b) => a - b);

  let pathColors = ["#0ff", "#f0f", "#ff0"];
  let defenders = [];
    // Collect defender positions from the board
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      if (typeof board[r][c] === "string") defenders.push([r, c, board[r][c]]);
    }
  }

   // Create attackers based on available defenders
  for (let i = 0; i < defenders.length; i++) {
    let col = usedCols[i];
    let spawn = [GRID_SIZE - 1, col];
    let chosenTarget = defenders[i];
    let pathType = Math.random() < 0.5 ? "straight" : "curve";
    let speed = Math.random() < 0.5 ? 1 : 2;
    let fullPath =
      pathType === "straight"
        ? straightPath(
            spawn[0],
            spawn[1],
            chosenTarget[0],
            chosenTarget[1]
          )
        : RandomPath(
            spawn[0],
            spawn[1],
            chosenTarget[0],
            chosenTarget[1]
          );
    // Ensure the final destination is exactly the chosen target
    if (
      fullPath[fullPath.length - 1][0] !== chosenTarget[0] ||
      fullPath[fullPath.length - 1][1] !== chosenTarget[1]
    ) {
      fullPath.push(chosenTarget);
    }
     // Reduce the full path based on the attacker's speed
    let steppedPath = [fullPath[0]];
    let currentIndex = 0;
    while (currentIndex < fullPath.length - 1) {
      let stepsRemaining = fullPath.length - 1 - currentIndex;
      let nextStep = Math.min(speed, stepsRemaining);
      currentIndex += nextStep;
      steppedPath.push(fullPath[currentIndex]);
    }
    // Add the attacker with its properties to the attackers array
    attackers.push({
      id: String.fromCharCode(65 + i),
      fullPath: fullPath,
      steppedPath: steppedPath,
      speed: speed,
      pathColor: pathColors[i],
      currentIndex: 0,
      baseTarget: chosenTarget,
    });
     // Initialize attacker history for movement logging
    attackerHistory[String.fromCharCode(65 + i)] = [
      [-1, -1],
      [-1, -1],
      [-1, -1],
      [-1, -1],
    ];
  }

  // If there are less defenders than attackers, create additional attackers with random target selection
  for (let i = defenders.length; i < 3; i++) {
    let col = usedCols[i];
    let spawn = [GRID_SIZE - 1, col];
    let chosenTarget = defenders[Math.floor(Math.random() * defenders.length)];
    let pathType = Math.random() < 0.5 ? "straight" : "curve";
    let speed = Math.random() < 0.5 ? 1 : 2;
    let fullPath =
      pathType === "straight"
        ? straightPath(
            spawn[0],
            spawn[1],
            chosenTarget[0],
            chosenTarget[1]
          )
        : RandomPath(
            spawn[0],
            spawn[1],
            chosenTarget[0],
            chosenTarget[1]
          );
    if (
      fullPath[fullPath.length - 1][0] !== chosenTarget[0] ||
      fullPath[fullPath.length - 1][1] !== chosenTarget[1]
    ) {
      fullPath.push(chosenTarget);
    }
    let steppedPath = [fullPath[0]];
    let currentIndex = 0;
    while (currentIndex < fullPath.length - 1) {
      let stepsRemaining = fullPath.length - 1 - currentIndex;
      let nextStep = Math.min(speed, stepsRemaining);
      currentIndex += nextStep;
      steppedPath.push(fullPath[currentIndex]);
    }
    attackers.push({
      id: String.fromCharCode(65 + i),
      fullPath: fullPath,
      steppedPath: steppedPath,
      speed: speed,
      pathColor: pathColors[i],
      currentIndex: 0,
      baseTarget: chosenTarget,
    });
    attackerHistory[String.fromCharCode(65 + i)] = [
      [-1, -1],
      [-1, -1],
      [-1, -1],
      [-1, -1],
    ];
  }
}

// Count the number of defenders currently on the board
function countDefenders() {
  let count = 0;
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      if (typeof board[r][c] === "string") count++;
    }
  }
  return count;
}

// Draw the game board, grid, and defenders on the canvas
function drawBoard(boardArr) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  ctx.font = "14px Arial";
  ctx.fillStyle = "#fff";
  ctx.strokeStyle = "#444";
  ctx.textAlign = "center";
  
   // Draw column numbers at the top of the grid
  for (let c = 0; c < GRID_SIZE; c++) {
    ctx.fillText(c.toString(), c * CELL_SIZE + 25 + CELL_SIZE / 2, 15);
  }
  
  // Draw row numbers on the left side of the grid
  ctx.textAlign = "right";
  for (let r = 0; r < GRID_SIZE; r++) {
    ctx.fillText(
      (GRID_SIZE - 1 - r).toString(),
      20,
      r * CELL_SIZE + 20 + CELL_SIZE / 2 + 5
    );
  }
  
  // Draw individual cells and defender images
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      ctx.strokeStyle = "#444";
      ctx.strokeRect(
        c * CELL_SIZE + 25,
        r * CELL_SIZE + 20,
        CELL_SIZE,
        CELL_SIZE
      );
      
      // If there is a defender in the cell, draw its image and label
      if (typeof boardArr[GRID_SIZE - 1 - r][c] === "string") {
        try {
          ctx.drawImage(
            defenderImg,
            c * CELL_SIZE + 30,
            r * CELL_SIZE + 25,
            CELL_SIZE - 10,
            CELL_SIZE - 10
          );
          
          ctx.fillStyle = "#fff";
          ctx.font = "bold 16px Arial";
          ctx.textAlign = "center";
          ctx.fillText(
            boardArr[GRID_SIZE - 1 - r][c],
            c * CELL_SIZE + 25 + CELL_SIZE/2,
            r * CELL_SIZE + 20 + CELL_SIZE/2 + 5
          );
        } catch (e) {
          console.error("Error drawing defender:", e);
        }
      }
    }
  }

  // Draw AI predictions if available
  if (window.predictions) {
    for (const prediction of window.predictions) {
      const attackerId = prediction.attackerId;
      const primaryPred = prediction.primary;
      const secondaryPred = prediction.secondary;

      // Draw primary prediction with attacker-specific color
      ctx.fillStyle = PREDICTION_COLORS[attackerId].primary.replace('{opacity}', primaryPred.confidence);
      ctx.fillRect(
        primaryPred.x * CELL_SIZE + 25,
        (GRID_SIZE - 1 - primaryPred.y) * CELL_SIZE + 20,
        CELL_SIZE,
        CELL_SIZE
      );
      
      // Draw secondary prediction with attacker-specific color
      ctx.fillStyle = PREDICTION_COLORS[attackerId].secondary.replace('{opacity}', secondaryPred.confidence);
      ctx.fillRect(
        secondaryPred.x * CELL_SIZE + 25,
        (GRID_SIZE - 1 - secondaryPred.y) * CELL_SIZE + 20,
        CELL_SIZE,
        CELL_SIZE
      );

      // Add prediction labels
      ctx.fillStyle = "#fff";
      ctx.font = "12px Arial";
      ctx.textAlign = "center";
      
      // Primary prediction label with exact percentage
      ctx.fillText(
        `${(primaryPred.confidence * 100).toFixed(1)}%`,
        primaryPred.x * CELL_SIZE + 25 + CELL_SIZE/2,
        (GRID_SIZE - 1 - primaryPred.y) * CELL_SIZE + 20 + CELL_SIZE/2
      );
      
      // Secondary prediction label with exact percentage
      ctx.fillText(
        `${(secondaryPred.confidence * 100).toFixed(1)}%`,
        secondaryPred.x * CELL_SIZE + 25 + CELL_SIZE/2,
        (GRID_SIZE - 1 - secondaryPred.y) * CELL_SIZE + 20 + CELL_SIZE/2
      );
    }
  }
  
  // Draw defender shots
  if (defenderShots["A"].length + defenderShots["B"].length > 0) {
    for (let defender in defenderShots) {
      defenderShots[defender].forEach(shot => {
        let [r, c] = shot;
        ctx.fillStyle = "rgba(172, 18, 172, 0.3)";
        ctx.fillRect(
          c * CELL_SIZE + 25,
          (GRID_SIZE - 1 - r) * CELL_SIZE + 20,
          CELL_SIZE,
          CELL_SIZE
        );
        
        ctx.fillStyle = "#FFFFFF";
        ctx.font = "bold 20px Arial";
        ctx.textAlign = "center";
        ctx.fillText(
          defender,
          c * CELL_SIZE + 25 + CELL_SIZE/2,
          (GRID_SIZE - 1 - r) * CELL_SIZE + 20 + CELL_SIZE/2 + 5
        );
      });
    }
  }
}


// Draw the paths for each attacker on the board
function drawPaths() {
  for (let atk of attackers) {
    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = atk.pathColor;
    ctx.lineWidth = 2;
    ctx.shadowColor = atk.pathColor;
    ctx.shadowBlur = 10;
    ctx.beginPath();
    for (let i = 0; i < atk.fullPath.length; i++) {
      let pr = atk.fullPath[i][0];
      let pc = atk.fullPath[i][1];
      let x = pc * CELL_SIZE + 25 + CELL_SIZE / 2;
      let y = (GRID_SIZE - 1 - pr) * CELL_SIZE + 20 + CELL_SIZE / 2; 
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;
  }
  ctx.setLineDash([]);
  ctx.font = "16px Arial";
  ctx.fillStyle = "#fff";
  ctx.textAlign = "center";
   // Label each step along the attacker path
  for (let atk of attackers) {
    for (let i = 1; i < atk.steppedPath.length; i++) {
      let pr = atk.steppedPath[i][0];
      let pc = atk.steppedPath[i][1];
      let x = pc * CELL_SIZE + 25 + CELL_SIZE / 2 - 5;
      let y = (GRID_SIZE - 1 - pr) * CELL_SIZE + 20 + CELL_SIZE / 2 + 5; 
      ctx.fillText(i.toString(), x, y);
    }
  }
}

function drawAttackers() {
  // Group attackers by their current tile
  const tileGroups = {};
  for (let atk of attackers) {
    let [r, c] = atk.steppedPath[atk.currentIndex];
    const key = `${r},${c}`;
    if (!tileGroups[key]) {
      tileGroups[key] = [];
    }
    tileGroups[key].push(atk);
  }

  // Iterate over each group and draw attackers with incremental offsets
  for (let key in tileGroups) {
    // Extract the row and column from the key
    const [r, c] = key.split(",").map(Number);
    // Set base drawing coordinates for the tile
    const baseX = c * CELL_SIZE + 30;
    const baseY = (GRID_SIZE - 1 - r) * CELL_SIZE + 25;
    const group = tileGroups[key];

    // Draw each attacker in the group, offsetting subsequent attackers by 5 pixels
    for (let i = 0; i < group.length; i++) {
      const atk = group[i];
      const offsetX = i * 5; // Fixed horizontal offset for staggered display
      const offsetY = i * 5; // Fixed vertical offset for staggered display
      
      try {
        // Draw the attacker image with the calculated offset
        ctx.drawImage(
          attackerImg,
          baseX + offsetX,
          baseY + offsetY,
          CELL_SIZE - 10,
          CELL_SIZE - 10
        );

        // Draw the attacker ID text at the center of the image with the same offset
        ctx.fillStyle = "#fff";
        ctx.font = "bold 16px Arial";
        ctx.textAlign = "center";
        ctx.fillText(
          atk.id,
          c * CELL_SIZE + 25 + CELL_SIZE / 2 + offsetX,
          (GRID_SIZE - 1 - r) * CELL_SIZE + 20 + CELL_SIZE / 2 + 5 + offsetY
        );
      } catch (e) {
        console.error("Error drawing attacker:", e);
      }
    }
  }
}


// Draw both the game board and the paths/attackers
function drawBoardAndPaths() {
  drawBoard(board);
  drawAttackers();
  if (showPaths) drawPaths();
}


// Initialize and start a new game
function newGame() {
  // Clear predictions
  window.predictions = null;
  const predictionOutput = document.getElementById('prediction-output');
  predictionOutput.style.display = 'none';
  predictionOutput.textContent = '';
  
  actions = [];
  actionLog.innerHTML = "";
  
  gameOver = false;
  statusMessage.textContent = "";
  nextTurnBtn.disabled = false;
  newGameBtn.disabled = false;
  showPaths = false;
  shotToggle = 0;
  board = createEmptyBoard();
  placeDefenders(board);
  placeAttackers();
  defenderShots = { A: [], B: [] }; // Start with empty shots
  hoveredCell = null;
  

  // Reset shot history for defenders
  defenderShotHistory = {
    A: [[-1,-1],[-1,-1],[-1,-1],[-1,-1]],
    B: [[-1,-1],[-1,-1],[-1,-1],[-1,-1]]
  };


  // Initialize attacker history with starting positions
  attackerHistory = {};
  for (let atk of attackers) { 
    let startPos = atk.steppedPath[0];
    attackerHistory[atk.id] = [
      [startPos[0], startPos[1]],
      [-1, -1],
      [-1, -1],
      [-1, -1]
    ];
  }
  
  updateDefenderShotHistory();
  
  // Save the initial board state for training data purposes
  trainingData.push(JSON.parse(JSON.stringify(board)));
  drawBoardAndPaths();
}


// End the game and display the reason
function endGame(reason) {
  gameOver = true;
  statusMessage.textContent = reason;
  actions.push("Game ended: " + reason);
  updateActionLog();
  
  if (!autoPlayActive) {
    nextTurnBtn.disabled = true;
  }
}


// Redirect attackers to new targets if their original defender target is destroyed
function redirectAttackers(destroyedDefender) {
  const remainingDefenders = [];
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      if (
        typeof board[r][c] === "string" &&
        board[r][c] !== destroyedDefender[2]
      ) {
        remainingDefenders.push([r, c, board[r][c]]);
      }
    }
  }
  if (remainingDefenders.length === 0) {
    drawBoardAndPaths();
    endGame("All defenders destroyed - Attackers win!");
    return;
  }
  // Update attackers that were targeting the destroyed defender
  for (let atk of attackers) {
    if (atk.baseTarget[2] === destroyedDefender[2]) {
      let newTarget = remainingDefenders[0];
      let minDist =
        Math.abs(newTarget[0] - atk.steppedPath[atk.currentIndex][0]) +
        Math.abs(newTarget[1] - atk.steppedPath[atk.currentIndex][1]);
      for (let def of remainingDefenders.slice(1)) {
        let dist =
          Math.abs(def[0] - atk.steppedPath[atk.currentIndex][0]) +
          Math.abs(def[1] - atk.steppedPath[atk.currentIndex][1]);
        if (dist < minDist) {
          minDist = dist;
          newTarget = def;
        }
      }
      atk.baseTarget = newTarget;
      let fullPath = straightPath(
        atk.steppedPath[atk.currentIndex][0],
        atk.steppedPath[atk.currentIndex][1],
        newTarget[0],
        newTarget[1]
      );
      if (
        fullPath[fullPath.length - 1][0] !== newTarget[0] ||
        fullPath[fullPath.length - 1][1] !== newTarget[1]
      ) {
        fullPath.push(newTarget);
      }
      let steppedPath = [fullPath[0]];
      let currentIndex = 0;
      while (currentIndex < fullPath.length - 1) {
        let stepsRemaining = fullPath.length - 1 - currentIndex;
        let nextStep = Math.min(atk.speed, stepsRemaining);
        currentIndex += nextStep;
        steppedPath.push(fullPath[currentIndex]);
      }
      atk.fullPath = fullPath;
      atk.steppedPath = steppedPath;
      atk.currentIndex = 0;
    }
  }
}

// Validate if a shot position is allowed (empty, no attacker present, and not already shot)
function isValidShotPosition(row, col) {
  return (
    row >= 0 && col >= 0 &&
    board[row][col] === 0 &&
    !attackers.some((a) => {
      let pos = a.steppedPath[a.currentIndex];
      return pos[0] === row && pos[1] === col;
    }) &&
    !Object.values(defenderShots)
      .flat()
      .some((t) => t[0] === row && t[1] === col)
  );
}


// Process and update prediction data from the server
function updatePredictions(serverResponse) {
  window.predictions = [];
  
  // For each attacker, update prediction objects with primary and secondary predictions
  for (const prediction of serverResponse) {
    const attackerId = prediction.attackerID;
    const [pred1x, pred1y, pred1conf, pred2x, pred2y, pred2conf] = prediction.predictions;
    
    window.predictions.push({
      attackerId,
      primary: {
        x: pred1x,
        y: pred1y,
        confidence: pred1conf
      },
      secondary: {
        x: pred2x,
        y: pred2y,
        confidence: pred2conf
      }
    });
  }
  
  // Redraw the board to show predictions
  drawBoardAndPaths();
}

function logAttackerData() {
  // Update prediction output box without changing its visibility
  predictionOutput.textContent = '\nReceived prediction request:\n';

  // Prepare and send logger data
  const logData = [];
  ['A', 'B', 'C'].forEach(id => {
    const history = attackerHistory[id];
    if (history[0][0] !== -1) { // Only alive attackers
      const positions = [];
      // Format: Oldest -> Newest as [T-3, T-2, T-1, Current]
      for (let i = history.length - 1; i >= 0; i--) {
        positions.push(history[i][1], history[i][0]); // col, row
      }
      logData.push({ attackerID: id, positions });
    }
  });

  fetch(LOGGER_SERVER, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(logData)
  })
  .then(res => {
    if (!res.ok) throw new Error('Network response was not ok');
    return res.json();
  })
  .then(predictions => {
    console.log("Received predictions:", predictions);
    
    // Clear existing predictions
    window.predictions = [];
    
    // Process each prediction
    predictions.forEach(pred => {
      const [pred1x, pred1y, pred1conf, pred2x, pred2y, pred2conf] = pred.predictions;
      
      // Add to window.predictions
      window.predictions.push({
        attackerId: pred.attackerID,
        primary: {
          x: pred1x,
          y: pred1y,
          confidence: pred1conf
        },
        secondary: {
          x: pred2x,
          y: pred2y,
          confidence: pred2conf
        }
      });

      // Update prediction output text
      predictionOutput.textContent += `\nAttacker ${pred.attackerID} prediction:\n`;
      const attackerData = logData.find(d => d.attackerID === pred.attackerID);
      predictionOutput.textContent += `T-2 Position: (${attackerData.positions[2]}, ${attackerData.positions[3]})\n`;
      predictionOutput.textContent += `T-1 Position: (${attackerData.positions[4]}, ${attackerData.positions[5]})\n`;
      predictionOutput.textContent += `Current Position: (${attackerData.positions[6]}, ${attackerData.positions[7]})\n`;
      predictionOutput.textContent += `Primary prediction: (${pred1x}, ${pred1y}) ${(pred1conf*100).toFixed(1)}%\n`;
      predictionOutput.textContent += `Secondary prediction: (${pred2x}, ${pred2y}) ${(pred2conf*100).toFixed(1)}%\n`;
    });
    
    predictionOutput.textContent += '';
    
    isLoggerConnected = true;
    // Force a redraw to show all predictions
    drawBoardAndPaths();
  })
  .catch(error => {
    console.error("Error getting predictions:", error);
    isLoggerConnected = false;
    window.predictions = null;
    predictionOutput.textContent += '\nError getting predictions\n';
    drawBoardAndPaths();
  });
}

function nextTurn() {
  if (gameOver) return;
  
  // Clear predictions when moving to next turn
  window.predictions = null;
  
  // Save current state of attackers and defenders for history tracking
  const preMoveState = {
    attackers: {},
    defenders: JSON.parse(JSON.stringify(defenderShots)),
  };

  attackers.forEach((atk) => {
    preMoveState.attackers[atk.id] = [...atk.steppedPath[atk.currentIndex]];
  });

  const movedAttackers = [];
  // Move each attacker one step along its path if possible
  attackers.forEach((atk) => {
    if (atk.currentIndex < atk.steppedPath.length - 1) {
      atk.currentIndex++;
      movedAttackers.push(atk);
      actions.push(
        `Attacker ${atk.id} moved to (${atk.steppedPath[atk.currentIndex][1]},${atk.steppedPath[atk.currentIndex][0]})`
      );
    }
  });

  const remainingAttackers = [];
  const destroyedDefenders = [];

  attackers.forEach((atk) => {
    const currentPos = atk.steppedPath[atk.currentIndex];
    let wasHit = false;

    // Check if the attacker has been hit by a shot
    Object.entries(defenderShots).forEach(([defender, shots]) => {
      if (
        shots.some(
          (shot) => shot[0] === currentPos[0] && shot[1] === currentPos[1]
        )
      ) {
        actions.push(
          `Defender ${defender} hit Attacker ${atk.id} at (${currentPos[1]},${currentPos[0]})`
        );
        wasHit = true;
      }
    });

    if (!wasHit) {
      // Check for defender collision at current position
      const currentDefender = board[currentPos[0]]?.[currentPos[1]];
      if (typeof currentDefender === "string") {
        // Destroy both attacker and defender
        board[currentPos[0]][currentPos[1]] = 0;
        destroyedDefenders.push([currentPos[0], currentPos[1], currentDefender]);
        actions.push(`Collision: Attacker ${atk.id} destroyed Defender ${currentDefender}`);
      }
      // Original target destruction logic
      else if (atk.currentIndex >= atk.steppedPath.length - 1) {
        const defenderPos = atk.baseTarget;
        const defender = board[defenderPos[0]]?.[defenderPos[1]];
        if (typeof defender === "string") {
          board[defenderPos[0]][defenderPos[1]] = 0;
          destroyedDefenders.push([...defenderPos, defender]);
          actions.push(`Attacker ${atk.id} destroyed Defender ${defender}`);
        }
      } else {
        remainingAttackers.push(atk);
      }
    }
  });

  // Redirect attackers for ALL destroyed defenders
  destroyedDefenders.forEach((defenderPos) => {
    redirectAttackers(defenderPos);
  });

  attackers = remainingAttackers;
  defenderShots = { A: [], B: [] }; // Reset shots
  updateAttackerHistory();
  updateDefenderShotHistory();
  
  drawBoardAndPaths();

  // End game conditions
  if (attackers.length === 0) {
    console.log("Ending game: All attackers destroyed");
    endGame("Defenders win!");
  }
  if (countDefenders() === 0) {
    console.log("Ending game: All defenders destroyed");
    endGame("Attackers win!");
  }

  updateActionLog();
}

function updateAttackerHistory() {
  // First create array of active attacker IDs
  const activeAttackerIds = attackers.map(atk => atk.id);

  // Update history for all possible attackers (A, B, C)
  ['A', 'B', 'C'].forEach(id => {
    if (!attackerHistory[id]) {
      attackerHistory[id] = [
        [-1, -1], [-1, -1], [-1, -1], [-1, -1]
      ];
    }

    // If attacker is active, update their position
    if (activeAttackerIds.includes(id)) {
      const attacker = attackers.find(atk => atk.id === id);
      const currentPos = attacker.steppedPath[attacker.currentIndex];
      
      // Only update if position changed
      if (currentPos[0] !== attackerHistory[id][0][0] || 
          currentPos[1] !== attackerHistory[id][0][1]) {
        attackerHistory[id].pop();
        attackerHistory[id].unshift([currentPos[0], currentPos[1]]);
      }
    } 
    // If attacker is not active, mark as destroyed
    else if (attackerHistory[id][0][0] !== -1 || attackerHistory[id][0][1] !== -1) {
      attackerHistory[id].pop();
      attackerHistory[id].unshift([-1, -1]);
    }
  });
}

// Update defender shot history arrays with the latest shot (or -1 if none)
function updateDefenderShotHistory() {
  Object.keys(defenderShots).forEach((defender) => {
    if (!defenderShotHistory[defender]) {
      defenderShotHistory[defender] = [
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
      ];
    }
    defenderShotHistory[defender].pop();
    defenderShotHistory[defender].unshift(
      defenderShots[defender][0] || [-1, -1]
    );
  });
}


// Create a separator element for the action log display
function createSeparator(character) {
  const separator = document.createElement('hr');
  separator.style.border = 'none';
  separator.style.borderTop = '1px dashed #444';
  separator.style.margin = '5px 0';
  
  if (character === '~') {
    separator.style.borderTop = '1px wavy #666';
  }
  
  return separator;
}


// Update the on-screen action log with messages from each turn
function updateActionLog() {
  actionLog.innerHTML = '';
  
  let currentGameStart = true;
  
  for (let i = 0; i < actions.length; i++) {
    const action = actions[i];
    
    // Add game separator if this is the first action of a new game
    if (currentGameStart) {
      actionLog.appendChild(createSeparator('~'));
      currentGameStart = false;
    }
    
    // Add action item
    const li = document.createElement('li');
    li.textContent = action;
    actionLog.appendChild(li);
    
    // Add turn separator if next action is from a different turn
    if (i < actions.length - 1 && 
        actions[i+1].includes("Defender") && 
        !action.includes("Defender")) {
      actionLog.appendChild(createSeparator('-'));
    }
    
    // Detect new game
    if (action.includes("Game ended")) {
      currentGameStart = true;
    }
  }
}

let shotToggle = 0;


// Handle canvas click events for defender shot selection
canvas.addEventListener("click", function(e) {
  if (gameOver || autoPlayActive) return;
  
  let rect = canvas.getBoundingClientRect();

  let scaleX = canvas.width / rect.width;
  let scaleY = canvas.height / rect.height;
  
  let x = (e.clientX - rect.left) * scaleX;
  let y = (e.clientY - rect.top) * scaleY;
  
  let offsetX = 25; 
  let offsetY = 20; 
  let col = Math.floor((x - offsetX) / CELL_SIZE);
  let row = GRID_SIZE - 1 - Math.floor((y - offsetY) / CELL_SIZE);
  
  if (col < 0 || col >= GRID_SIZE || row < 0 || row >= GRID_SIZE) return;
  
  hoveredCell = [row, col];
  if (!isValidShotPosition(row, col)) return;
  

   // Alternate shot selection between defender A and B
  let defender = (shotToggle % 2 === 0) ? "A" : "B";
  shotToggle++;
  
  defenderShots[defender] = [[row, col]];
  defenderShotHistory[defender][0] = defenderShots[defender][0];
  actions.push(
    "Defender " + defender + " selected shot at (" + col + "," + row + ")"
  );
  
  updateActionLog();
  drawBoardAndPaths();
});

// Toggle the legend visibility on the board when the related button is clicked
const toggleLegendBtn = document.getElementById("toggleLegendBtn");
const mapLegend = document.getElementById("mapLegend");

toggleLegendBtn.addEventListener("click", function() {
  // Toggle the display style of the legend element
  if (mapLegend.style.display === "none") {
    mapLegend.style.display = "block";
  } else {
    mapLegend.style.display = "none";
  }
});


// Attach event listeners to UI buttons for various actions
document.getElementById('makePredictionsBtn').addEventListener('click', logAttackerData);
newGameBtn.addEventListener("click", newGame);
nextTurnBtn.addEventListener("click", nextTurn);
actionLogBtn.addEventListener("click", function () {
  actionLog.style.display =
    actionLog.style.display === "none" ? "block" : "none";
});
togglePathsBtn.addEventListener("click", function () {
  showPaths = !showPaths;
  drawBoardAndPaths();
});
togglePredictionOutputBtn.addEventListener("click", function() {
  predictionOutput.style.display = predictionOutput.style.display === "none" ? "block" : "none";
});
autoPlayBtn.addEventListener("click", toggleAutoPlay);


// Get a list of current defender positions from the board
function getCurrentDefenders() {
    let defenders = [];
    for (let r = 0; r < GRID_SIZE; r++) {
        for (let c = 0; c < GRID_SIZE; c++) {
            if (typeof board[r][c] === "string") {
                defenders.push([r, c, board[r][c]]);
            }
        }
    }
    return defenders;
}


// Draw the prediction board (a smaller canvas representation) showing predicted positions and current positions
function drawPredictionCanvas(canvas, board, attackers, defenders, defenderShots, turnOffset = 0) {
    const ctx = canvas.getContext('2d');
    const PRED_CELL_SIZE = 36;
    

    // Fill the prediction canvas with a dark background
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw grid lines on the prediction canvas
    ctx.strokeStyle = '#444';
    ctx.lineWidth = 1;
    for (let i = 0; i <= GRID_SIZE; i++) {
        ctx.beginPath();
        ctx.moveTo(i * PRED_CELL_SIZE + 25, 20);
        ctx.lineTo(i * PRED_CELL_SIZE + 25, PRED_CELL_SIZE * GRID_SIZE + 20);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(25, i * PRED_CELL_SIZE + 20);
        ctx.lineTo(PRED_CELL_SIZE * GRID_SIZE + 25, i * PRED_CELL_SIZE + 20);
        ctx.stroke();
    }
    

    // Label grid rows and columns
    ctx.fillStyle = '#666';
    ctx.font = '10px Arial';
    ctx.textAlign = 'center';
    for (let i = 0; i < GRID_SIZE; i++) {
        ctx.fillText(i.toString(), i * PRED_CELL_SIZE + 25 + PRED_CELL_SIZE/2, 15);
        ctx.textAlign = 'right';
        ctx.fillText((GRID_SIZE - 1 - i).toString(), 20, i * PRED_CELL_SIZE + 20 + PRED_CELL_SIZE/2);
    }
    


    // Draw defenders on the prediction canvas
    for (let r = 0; r < GRID_SIZE; r++) {
        for (let c = 0; c < GRID_SIZE; c++) {
            if (typeof board[r][c] === "string") {
                try {
                    ctx.drawImage(
                        defenderImg,
                        c * PRED_CELL_SIZE + 27,
                        (GRID_SIZE - 1 - r) * PRED_CELL_SIZE + 22,
                        PRED_CELL_SIZE - 4,
                        PRED_CELL_SIZE - 4
                    );
                    
                    ctx.fillStyle = "white";
                    ctx.font = "bold 14px Arial";
                    ctx.textAlign = "center";
                    ctx.fillText(
                        board[r][c],
                        c * PRED_CELL_SIZE + 25 + PRED_CELL_SIZE/2,
                        (GRID_SIZE - 1 - r) * PRED_CELL_SIZE + 20 + PRED_CELL_SIZE/2 + 5
                    );
                } catch (e) {
                    console.error("Error drawing defender in prediction:", e);
                }
            }
        }
    }
    

    // Draw defender shots on the prediction canvas
    Object.entries(defenderShots).forEach(([defender, shots]) => {
        shots.forEach(shot => {
            const [r, c] = shot;
            ctx.fillStyle = "rgba(172, 18, 172, 0.3)";
            ctx.fillRect(
                c * PRED_CELL_SIZE + 25,
                (GRID_SIZE - 1 - r) * PRED_CELL_SIZE + 20,
                PRED_CELL_SIZE,
                PRED_CELL_SIZE
            );
            
            ctx.fillStyle = "#FFFFFF";
            ctx.font = "bold 16px Arial";
            ctx.textAlign = "center";
            ctx.fillText(
                defender,
                c * PRED_CELL_SIZE + 25 + PRED_CELL_SIZE/2,
                (GRID_SIZE - 1 - r) * PRED_CELL_SIZE + 20 + PRED_CELL_SIZE/2 + 5
            );
        });
    });
    

    // Draw each attacker on the prediction canvas at their predicted positions
    attackers.forEach(attacker => {
        let predictedIndex = Math.min(attacker.currentIndex + turnOffset, attacker.steppedPath.length - 1);
        let [r, c] = attacker.steppedPath[predictedIndex];
        
        try {
            ctx.drawImage(
                attackerImg,
                c * PRED_CELL_SIZE + 27,
                (GRID_SIZE - 1 - r) * PRED_CELL_SIZE + 22,
                PRED_CELL_SIZE - 4,
                PRED_CELL_SIZE - 4
            );
            
            ctx.fillStyle = 'white';
            ctx.font = 'bold 14px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(
                attacker.id,
                c * PRED_CELL_SIZE + 25 + PRED_CELL_SIZE/2,
                (GRID_SIZE - 1 - r) * PRED_CELL_SIZE + 20 + PRED_CELL_SIZE/2 + 5
            );
        } catch (e) {
            console.error("Error drawing attacker in prediction:", e);
        }
    });
}


// Generate possible shot positions based on attacker positions, using two strategies
function generatePossibleShots() {
    const possibleShots = [];
    
    // Get current defender and attacker positions
    const defenders = getCurrentDefenders();
    const currentAttackerPositions = attackers.map(atk => ({
        id: atk.id,
        pos: atk.steppedPath[atk.currentIndex],
        nextPos: atk.steppedPath[Math.min(atk.currentIndex + 1, atk.steppedPath.length - 1)]
    }));

    // Best prediction: Try to shoot where attackers will be next turn
    const bestPrediction = { A: [], B: [] };
    defenders.forEach((defender, idx) => {
        if (currentAttackerPositions[idx]) {
            bestPrediction[defender[2]] = [currentAttackerPositions[idx].nextPos];
        }
    });
    possibleShots.push(bestPrediction);

    // Second best prediction: Shoot one square ahead of current attacker positions
    const secondBestPrediction = { A: [], B: [] };
    defenders.forEach((defender, idx) => {
        if (currentAttackerPositions[idx]) {
            const currentPos = currentAttackerPositions[idx].pos;
            const nextPos = [
                Math.min(currentPos[0] + 1, GRID_SIZE - 1),
                currentPos[1]
            ];
            secondBestPrediction[defender[2]] = [nextPos];
        }
    });
    possibleShots.push(secondBestPrediction);

    return possibleShots;
}


// Evaluate the quality of a given shot prediction against attacker positions
function evaluatePrediction(shots) {
    let score = 0;
    let maxPossibleScore = attackers.length * 100; // Perfect score if all shots are direct hits
    
    attackers.forEach(atk => {
        const nextPos = atk.steppedPath[Math.min(atk.currentIndex + 1, atk.steppedPath.length - 1)];
        
        Object.entries(shots).forEach(([defender, defenderShots]) => {
            defenderShots.forEach(shot => {
                // Direct hit
                if (shot[0] === nextPos[0] && shot[1] === nextPos[1]) {
                    score += 100;
                }
                // Near miss
                else if (Math.abs(shot[0] - nextPos[0]) + Math.abs(shot[1] - nextPos[1]) === 1) {
                    score += 50;
                }
                // Distance penalty
                else {
                    score -= 2 * (Math.abs(shot[0] - nextPos[0]) + Math.abs(shot[1] - nextPos[1]));
                }
            });
        });
    });
    
    // Convert to score out of 10
    return Math.max(0, Math.min(10, (score / maxPossibleScore) * 10));
}
