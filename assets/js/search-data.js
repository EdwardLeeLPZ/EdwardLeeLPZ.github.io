// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-home",
    title: "Home",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-publications",
          title: "Publications",
          description: "Check the latest through Google Scholar.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-blogs",
          title: "Blogs",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-cv",
          title: "CV",
          description: "Education, work experience, technical skills, and academic service.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "news-powerbev-our-paper-on-camera-based-end-to-end-instance-prediction-in-bird-s-eye-view-has-been-accepted-by-ijcai-2023",
          title: 'PowerBEV, our paper on camera-based end-to-end instance prediction in bird’s-eye view, has been...',
          description: "",
          section: "News",},{id: "news-our-seflow-a-self-supervised-scene-flow-method-in-autonomous-driving-paper-is-accepted-by-eccv-2024-the-1st-ranking-on-argoverse-2-self-supervised-scene-flow-leaderboard",
          title: 'Our SeFlow: A Self-Supervised Scene Flow Method in Autonomous Driving paper is accepted...',
          description: "",
          section: "News",},{id: "news-our-ago-adaptive-grounding-for-open-world-3d-occupancy-prediction-paper-is-accepted-by-iccv-2025",
          title: 'Our AGO: Adaptive Grounding for Open World 3D Occupancy Prediction paper is accepted...',
          description: "",
          section: "News",},{id: "news-our-spacedrive-infusing-spatial-awareness-into-vlm-based-autonomous-driving-paper-is-accepted-by-cvpr-2026-the-1st-ranking-on-nuscenes-benchmark-and-2nd-best-close-loop-performance-on-bench2drive-leaderboard",
          title: 'Our SpaceDrive: Infusing Spatial Awareness into VLM-based Autonomous Driving paper is accepted by...',
          description: "",
          section: "News",},{id: "news-two-papers-g2dp-diffusion-planning-with-spatio-temporal-grid-guidance-and-shift-amp-amp-drift-a-zero-shot-benchmark-for-generalizable-and-robust-autonomous-driving-motion-planning-are-accepted-by-iros-2026",
          title: 'Two papers, G2DP: Diffusion Planning with Spatio-Temporal Grid Guidance and Shift &amp;amp;amp; Drift:...',
          description: "",
          section: "News",},{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
