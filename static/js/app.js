// Cricket Face Identification AI - Frontend Controller
document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const selectPhotoBtn = document.getElementById("selectPhotoBtn");
  const previewImage = document.getElementById("previewImage");
  const bboxLayer = document.getElementById("bboxLayer");
  const imageStageWrap = document.getElementById("imageStageWrap");
  const viewerEmptyState = document.getElementById("viewerEmptyState");
  const viewerContainer = document.getElementById("viewerContainer");
  const detectionStatusMsg = document.getElementById("detectionStatusMsg");
  const detectionCountBadge = document.getElementById("detectionCountBadge");
  const prevFaceBtn = document.getElementById("prevFaceBtn");
  const nextFaceBtn = document.getElementById("nextFaceBtn");
  const visualSearchBtn = document.getElementById("visualSearchBtn");
  const loadingOverlay = document.getElementById("loadingOverlay");

  // Tabs
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  // Camera
  const webcamVideo = document.getElementById("webcamVideo");
  const cameraCanvas = document.getElementById("cameraCanvas");
  const snapPhotoBtn = document.getElementById("snapPhotoBtn");
  const toggleCameraBtn = document.getElementById("toggleCameraBtn");
  let cameraStream = null;

  // Quick Stars
  const quickStarsGrid = document.getElementById("quickStarsGrid");

  // Player Dossier Elements
  const playerAvatar = document.getElementById("playerAvatar");
  const playerRoleBadge = document.getElementById("playerRoleBadge");
  const playerNameHeading = document.getElementById("playerNameHeading");
  const playerEraText = document.getElementById("playerEraText");
  const confidenceValue = document.getElementById("confidenceValue");
  const progressBarFill = document.getElementById("progressBarFill");
  const lowConfidenceAlert = document.getElementById("lowConfidenceAlert");
  const battingStyleVal = document.getElementById("battingStyleVal");
  const bowlingStyleVal = document.getElementById("bowlingStyleVal");
  const careerMatchesVal = document.getElementById("careerMatchesVal");
  const keyTeamsVal = document.getElementById("keyTeamsVal");
  const bestPerformanceVal = document.getElementById("bestPerformanceVal");
  const highlightsList = document.getElementById("highlightsList");
  const biographyVal = document.getElementById("biographyVal");
  const browseLegendsLink = document.getElementById("browseLegendsLink");

  // Roster Modal Elements
  const rosterModal = document.getElementById("rosterModal");
  const openRosterBtn = document.getElementById("openRosterBtn");
  const closeRosterBtn = document.getElementById("closeRosterBtn");
  const rosterSearchInput = document.getElementById("rosterSearchInput");
  const modalRosterList = document.getElementById("modalRosterList");
  const filterPills = document.querySelectorAll(".filter-pill");

  // State
  let currentDetections = [];
  let currentActiveFaceIndex = 0;
  let allRosterPlayers = [];
  let currentFilter = "all";
  let activeImageNaturalSize = { width: 1, height: 1 };

  // -----------------------------------------------------------------------
  // 1. Tab Navigation
  // -----------------------------------------------------------------------
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");
      tabBtns.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPane = document.getElementById(targetTab);
      if (targetPane) targetPane.classList.add("active");

      if (targetTab !== "cameraTab" && cameraStream) {
        stopCamera();
      }
    });
  });

  // -----------------------------------------------------------------------
  // 2. Drag & Drop and File Selection
  // -----------------------------------------------------------------------
  selectPhotoBtn.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("click", (e) => {
    if (e.target !== selectPhotoBtn) fileInput.click();
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
    }, false);
  });

  dropZone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      handleImageFile(files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleImageFile(e.target.files[0]);
    }
  });

  function showActiveImage() {
    if (viewerEmptyState) viewerEmptyState.style.display = "none";
    if (imageStageWrap) imageStageWrap.style.display = "inline-block";
    previewImage.style.display = "block";
  }

  function handleImageFile(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please upload an image file (JPG, PNG, or WEBP).");
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      previewImage.onload = () => {
        showActiveImage();
        activeImageNaturalSize = {
          width: previewImage.naturalWidth,
          height: previewImage.naturalHeight
        };
        uploadAndPredict(file);
      };
    };
    reader.readAsDataURL(file);
  }

  // -----------------------------------------------------------------------
  // 3. API Prediction Request
  // -----------------------------------------------------------------------
  async function uploadAndPredict(fileOrBlob) {
    showLoading(true);
    clearBoundingBoxes();
    detectionStatusMsg.textContent = "Analyzing face embeddings against 324 legends...";

    const formData = new FormData();
    formData.append("image", fileOrBlob);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData
      });

      const data = await response.json();
      if (!response.ok || !data.success) {
        throw new Error(data.error || "Detection failed");
      }

      renderDetections(data);
    } catch (err) {
      console.error("Prediction error:", err);
      detectionStatusMsg.textContent = "Error: " + err.message;
      alert("Identification error: " + err.message);
    } finally {
      showLoading(false);
    }
  }

  // -----------------------------------------------------------------------
  // 4. Render Bounding Boxes & Multi-Face Carousel
  // -----------------------------------------------------------------------
  function renderDetections(data) {
    currentDetections = data.detections || [];
    detectionCountBadge.textContent = currentDetections.length;
    clearBoundingBoxes();

    if (currentDetections.length === 0) {
      detectionStatusMsg.textContent = "⚠️ No human face detected in the image.";
      prevFaceBtn.disabled = true;
      nextFaceBtn.disabled = true;
      resetDossierToNoFace(data.message);
      return;
    }

    detectionStatusMsg.textContent = `Identified ${currentDetections.length} face${currentDetections.length > 1 ? 's' : ''}. Select a face box or use arrows to view profile.`;

    prevFaceBtn.disabled = currentDetections.length <= 1;
    nextFaceBtn.disabled = currentDetections.length <= 1;

    const imgRect = previewImage.getBoundingClientRect();
    const naturalW = data.image_size ? data.image_size[0] : activeImageNaturalSize.width;
    const naturalH = data.image_size ? data.image_size[1] : activeImageNaturalSize.height;

    currentDetections.forEach((det, idx) => {
      const [x, y, w, h] = det.box;

      const leftPct = (x / naturalW) * 100;
      const topPct = (y / naturalH) * 100;
      const widthPct = (w / naturalW) * 100;
      const heightPct = (h / naturalH) * 100;

      const boxElem = document.createElement("div");
      boxElem.className = `bounding-box ${idx === 0 ? "active-box" : ""}`;
      boxElem.id = `bbox-${idx}`;
      boxElem.style.left = `${leftPct}%`;
      boxElem.style.top = `${topPct}%`;
      boxElem.style.width = `${widthPct}%`;
      boxElem.style.height = `${heightPct}%`;

      const labelElem = document.createElement("div");
      labelElem.className = "bbox-label";
      labelElem.textContent = `${det.player_name} (${det.confidence}%)`;

      boxElem.appendChild(labelElem);
      boxElem.addEventListener("click", () => setActiveFace(idx));

      bboxLayer.appendChild(boxElem);
    });

    // Set first detection as active
    currentActiveFaceIndex = 0;
    setActiveFace(0);
  }

  function resetDossierToNoFace(customMsg) {
    playerRoleBadge.textContent = "NO FACE";
    playerRoleBadge.style.background = "rgba(255, 71, 87, 0.15)";
    playerRoleBadge.style.color = "#ff6b81";
    playerRoleBadge.style.borderColor = "rgba(255, 71, 87, 0.35)";

    playerNameHeading.textContent = "NO CRICKETER FACE DETECTED";
    playerEraText.textContent = "No face was recognized in the uploaded image";

    confidenceValue.textContent = "0.0%";
    progressBarFill.style.width = "0%";

    lowConfidenceAlert.style.display = "flex";
    lowConfidenceAlert.querySelector(".alert-body").innerHTML = `
      <strong>No Face Detected</strong> — ${customMsg || "The AI scanner did not find any human faces in this image."}
      <p style="margin-top: 4px;">Please upload an image with a visible cricketer face, or click any player in <strong>Quick Test Stars</strong>.</p>
    `;

    battingStyleVal.textContent = "—";
    bowlingStyleVal.textContent = "—";
    careerMatchesVal.textContent = "—";
    keyTeamsVal.textContent = "—";

    bestPerformanceVal.textContent = "No face was detected. Please upload a clear photo containing a cricketer's face.";

    highlightsList.innerHTML = `
      <li><span class="trophy-emoji">🔍</span> <span>No human faces were found by the neural network in this uploaded image.</span></li>
      <li><span class="trophy-emoji">🏏</span> <span>Ensure the photo has good lighting and an unobstructed face.</span></li>
      <li><span class="trophy-emoji">⭐</span> <span>Tip: You can use the Quick Test Stars tab to test preloaded star cricketers.</span></li>
    `;

    biographyVal.textContent = "The MTCNN face detector evaluated this image and determined that no human facial features exist here (e.g. document, screenshot, or non-face object). Please upload a photo with an Indian cricketer's face.";

    playerAvatar.src = "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><circle cx='50' cy='50' r='48' fill='%2313233f'/><text x='50' y='60' font-size='38' text-anchor='middle' fill='%23ff6b81'>❓</text></svg>";
  }

  function resetDossierToReady() {
    playerRoleBadge.textContent = "READY";
    playerRoleBadge.style.background = "rgba(0, 229, 153, 0.15)";
    playerRoleBadge.style.color = "var(--accent-green)";
    playerRoleBadge.style.borderColor = "rgba(0, 229, 153, 0.35)";

    playerNameHeading.textContent = "SELECT OR UPLOAD A PLAYER";
    playerEraText.textContent = "Active Era: Database Ready (324 Cricketers)";

    confidenceValue.textContent = "0.0%";
    progressBarFill.style.width = "0%";
    lowConfidenceAlert.style.display = "none";

    battingStyleVal.textContent = "—";
    bowlingStyleVal.textContent = "—";
    careerMatchesVal.textContent = "—";
    keyTeamsVal.textContent = "—";

    bestPerformanceVal.textContent = "Upload a cricketer image or select a Quick Test Star to inspect career records.";

    highlightsList.innerHTML = `
      <li><span class="trophy-emoji">🏆</span> <span>Upload a photo or select a Quick Test Star to generate intelligence dossier.</span></li>
    `;

    biographyVal.textContent = "Our deep neural network identifies Indian cricket legends, modern champions, and historic players across Tests, ODIs, T20Is, and domestic championships using deep metric facial embeddings.";

    playerAvatar.src = "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><circle cx='50' cy='50' r='48' fill='%2313233f'/><text x='50' y='60' font-size='38' text-anchor='middle' fill='%2300e599'>🏏</text></svg>";
  }

  function clearBoundingBoxes() {
    bboxLayer.innerHTML = "";
  }

  function setActiveFace(idx) {
    if (idx < 0 || idx >= currentDetections.length) return;
    currentActiveFaceIndex = idx;

    // Highlight active bounding box
    document.querySelectorAll(".bounding-box").forEach((b, i) => {
      if (i === idx) {
        b.classList.add("active-box");
      } else {
        b.classList.remove("active-box");
      }
    });

    const activeDetection = currentDetections[idx];
    updateDossier(activeDetection);
  }

  prevFaceBtn.addEventListener("click", () => {
    if (currentDetections.length <= 1) return;
    const newIdx = (currentActiveFaceIndex - 1 + currentDetections.length) % currentDetections.length;
    setActiveFace(newIdx);
  });

  nextFaceBtn.addEventListener("click", () => {
    if (currentDetections.length <= 1) return;
    const newIdx = (currentActiveFaceIndex + 1) % currentDetections.length;
    setActiveFace(newIdx);
  });

  // -----------------------------------------------------------------------
  // 5. Update Player Dossier Panel
  // -----------------------------------------------------------------------
  function updateDossier(detection) {
    const d = detection.dossier;

    // Crop face avatar for the profile circle
    cropFaceAvatar(detection);

    playerRoleBadge.textContent = d.role_badge || "CRICKETER";
    playerNameHeading.textContent = d.display_name.toUpperCase();
    playerEraText.textContent = `Active Era: ${d.active_era || "International & Domestic Circuit"}`;

    // Confidence
    const conf = detection.confidence;
    confidenceValue.textContent = `${conf.toFixed(1)}%`;
    progressBarFill.style.width = `${Math.min(100, Math.max(5, conf))}%`;

    // Low confidence alert (Screenshot 3)
    if (detection.is_low_confidence || conf < 45.0) {
      lowConfidenceAlert.style.display = "flex";
    } else {
      lowConfidenceAlert.style.display = "none";
    }

    // Specs Grid
    battingStyleVal.textContent = d.batting_style || "Right-hand bat";
    bowlingStyleVal.textContent = d.bowling_style || "Right-arm medium";
    careerMatchesVal.textContent = d.career_matches || "Tests, ODIs, T20Is";
    keyTeamsVal.textContent = d.key_teams || "India";

    // Best Performance
    bestPerformanceVal.textContent = d.best_performance || "Distinguished match-winning contributions in international encounters.";

    // Highlights List
    highlightsList.innerHTML = "";
    const highlights = d.career_highlights || [
      "Key match-winning performer for the Indian national squad",
      "Celebrated across competitive international fixtures"
    ];
    highlights.forEach(h => {
      const li = document.createElement("li");
      li.innerHTML = `<span class="trophy-emoji">🏆</span> <span>${h}</span>`;
      highlightsList.appendChild(li);
    });

    // Biography
    biographyVal.textContent = d.biography || `${d.display_name} is an elite member of the 324-player Indian cricket repository.`;
  }

  function cropFaceAvatar(detection) {
    try {
      const [x, y, w, h] = detection.box;
      const canvas = document.createElement("canvas");
      canvas.width = 160;
      canvas.height = 160;
      const ctx = canvas.getContext("2d");

      ctx.drawImage(previewImage, x, y, w, h, 0, 0, 160, 160);
      playerAvatar.src = canvas.toDataURL("image/jpeg", 0.9);
    } catch (e) {
      playerAvatar.src = previewImage.src;
    }
  }

  // -----------------------------------------------------------------------
  // 6. Live Camera
  // -----------------------------------------------------------------------
  toggleCameraBtn.addEventListener("click", () => {
    if (cameraStream) {
      stopCamera();
    } else {
      startCamera();
    }
  });

  async function startCamera() {
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }
      });
      webcamVideo.srcObject = cameraStream;
      toggleCameraBtn.textContent = "Stop Camera";
    } catch (err) {
      alert("Unable to access webcam: " + err.message);
    }
  }

  function stopCamera() {
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      cameraStream = null;
      webcamVideo.srcObject = null;
      toggleCameraBtn.textContent = "Start Camera";
    }
  }

  snapPhotoBtn.addEventListener("click", () => {
    if (!cameraStream) {
      alert("Please start the camera first.");
      return;
    }

    cameraCanvas.width = webcamVideo.videoWidth || 640;
    cameraCanvas.height = webcamVideo.videoHeight || 480;
    const ctx = cameraCanvas.getContext("2d");
    ctx.drawImage(webcamVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);

    cameraCanvas.toBlob((blob) => {
      if (blob) {
        previewImage.src = URL.createObjectURL(blob);
        previewImage.onload = () => {
          showActiveImage();
          activeImageNaturalSize = {
            width: cameraCanvas.width,
            height: cameraCanvas.height
          };
          uploadAndPredict(blob);
        };
      }
    }, "image/jpeg", 0.95);
  });

  // -----------------------------------------------------------------------
  // 7. Quick Test Stars
  // -----------------------------------------------------------------------
  async function loadQuickStars() {
    try {
      const res = await fetch("/api/samples");
      const data = await res.json();
      quickStarsGrid.innerHTML = "";

      data.samples.forEach(star => {
        const card = document.createElement("div");
        card.className = "star-card";
        card.innerHTML = `
          <img src="/static/samples/${star.file}" class="star-card-img" alt="${star.name}">
          <span class="star-card-name">${star.name}</span>
          <span class="star-card-badge">${star.badge}</span>
        `;
        card.addEventListener("click", () => selectQuickStar(star, card));
        quickStarsGrid.appendChild(card);
      });
    } catch (err) {
      quickStarsGrid.innerHTML = `<div class="error-stars">Failed to load quick test stars.</div>`;
    }
  }

  async function selectQuickStar(star, cardElem) {
    document.querySelectorAll(".star-card").forEach(c => c.classList.remove("active"));
    cardElem.classList.add("active");

    const sampleImgUrl = `/static/samples/${star.file}`;
    previewImage.src = sampleImgUrl;

    previewImage.onload = async () => {
      showActiveImage();
      activeImageNaturalSize = {
        width: previewImage.naturalWidth,
        height: previewImage.naturalHeight
      };

      showLoading(true);
      clearBoundingBoxes();
      detectionStatusMsg.textContent = `Running identification on sample: ${star.name}...`;

      try {
        const response = await fetch(`/api/predict?sample=${encodeURIComponent(star.file)}`, {
          method: "POST"
        });
        const data = await response.json();
        renderDetections(data);
      } catch (e) {
        alert("Error testing sample: " + e.message);
      } finally {
        showLoading(false);
      }
    };
  }

  // -----------------------------------------------------------------------
  // 8. Legends Roster Modal (324 Cricketers)
  // -----------------------------------------------------------------------
  openRosterBtn.addEventListener("click", () => openRoster());
  browseLegendsLink.addEventListener("click", (e) => {
    e.preventDefault();
    openRoster();
  });
  closeRosterBtn.addEventListener("click", () => closeRoster());

  rosterModal.addEventListener("click", (e) => {
    if (e.target === rosterModal) closeRoster();
  });

  async function openRoster() {
    rosterModal.classList.add("open");
    if (allRosterPlayers.length === 0) {
      await fetchRoster();
    }
    renderRosterList();
  }

  function closeRoster() {
    rosterModal.classList.remove("open");
  }

  async function fetchRoster() {
    try {
      const res = await fetch("/api/players");
      const data = await res.json();
      allRosterPlayers = data.players || [];
    } catch (e) {
      console.error("Failed to fetch roster:", e);
    }
  }

  function renderRosterList() {
    modalRosterList.innerHTML = "";
    const query = rosterSearchInput.value.toLowerCase().trim();

    const filtered = allRosterPlayers.filter(p => {
      const matchesQuery = p.display_name.toLowerCase().includes(query) ||
                           p.role.toLowerCase().includes(query) ||
                           p.teams.toLowerCase().includes(query);
      const matchesFilter = (currentFilter === "all") || (p.role_badge === currentFilter);
      return matchesQuery && matchesFilter;
    });

    if (filtered.length === 0) {
      modalRosterList.innerHTML = `<div style="grid-column: 1/-1; text-align: center; color: var(--text-muted); padding: 40px;">No cricketers match your search.</div>`;
      return;
    }

    filtered.forEach(p => {
      const item = document.createElement("div");
      item.className = "roster-player-item";
      item.innerHTML = `
        <div class="roster-item-info">
          <h4>${p.display_name}</h4>
          <p>${p.role} • ${p.batting_style || 'Bat'}</p>
        </div>
        <span class="role-pill-badge" style="font-size: 0.65rem;">${p.role_badge}</span>
      `;
      item.addEventListener("click", () => {
        closeRoster();
        // Set mock detection to load dossier
        const mockDetection = {
          confidence: 96.5,
          box: [0, 0, activeImageNaturalSize.width, activeImageNaturalSize.height],
          is_low_confidence: false,
          dossier: p
        };
        updateDossier(mockDetection);
      });
      modalRosterList.appendChild(item);
    });
  }

  rosterSearchInput.addEventListener("input", () => renderRosterList());

  filterPills.forEach(pill => {
    pill.addEventListener("click", () => {
      filterPills.forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      currentFilter = pill.getAttribute("data-filter");
      renderRosterList();
    });
  });

  // Visual Search Button Animation
  visualSearchBtn.addEventListener("click", () => {
    if (currentDetections.length > 0) {
      setActiveFace((currentActiveFaceIndex + 1) % currentDetections.length);
    }
  });

  function showLoading(show) {
    if (show) {
      loadingOverlay.classList.add("active");
    } else {
      loadingOverlay.classList.remove("active");
    }
  }

  // Initial Load
  loadQuickStars();
});

