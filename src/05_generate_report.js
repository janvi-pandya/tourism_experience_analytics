const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, ImageRun, AlignmentType, BorderStyle, PageBreak
} = require("docx");

const FIG = "outputs/figures";

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 250, after: 120 } });
}
function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 120 } });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 60 } });
}
function image(path, width = 500, height = 280) {
  const data = fs.readFileSync(path);
  return new Paragraph({
    children: [new ImageRun({ data, transformation: { width, height }, type: "png" })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
  });
}
function cell(text, opts = {}) {
  return new TableCell({
    children: [new Paragraph({ children: [new TextRun({ text, bold: !!opts.bold })] })],
    width: { size: opts.width || 25, type: WidthType.PERCENTAGE },
    shading: opts.bold ? { type: ShadingType.CLEAR, fill: "D9E2F3" } : undefined,
  });
}
function metricsTable(rows) {
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    rows: [
      new TableRow({ children: [cell("Metric", { bold: true, width: 40 }), cell("Value", { bold: true, width: 60 })] }),
      ...rows.map(([m, v]) => new TableRow({ children: [cell(m, { width: 40 }), cell(v, { width: 60 })] })),
    ],
  });
}

const doc = new Document({
  sections: [
    {
      properties: { page: { size: { width: 12240, height: 15840 } } },
      children: [
        new Paragraph({
          children: [new TextRun({ text: "Tourism Experience Analytics", bold: true, size: 44 })],
          alignment: AlignmentType.CENTER, spacing: { after: 100 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "Project Documentation Report", size: 28, color: "555555" })],
          alignment: AlignmentType.CENTER, spacing: { after: 400 },
        }),

        h1("1. Project Overview"),
        p("This project builds an end-to-end analytics pipeline for a tourism platform, combining transactional visit data with user and attraction metadata to power three capabilities: predicting how a user will rate an attraction (regression), predicting how a user is likely to travel (classification of visit mode), and recommending attractions a user is likely to enjoy (collaborative filtering with a content-based fallback). The pipeline was deployed as an interactive Streamlit application."),

        h1("2. Data Pipeline"),
        p("Ten relational tables were merged into a single master dataset: Transaction, User, City, Country, Region, Continent, Visit Mode, Attraction Type, and Attraction (Item) tables, including an expanded attraction catalog (1,698 attractions vs. the original 30-row file)."),
        h2("2.1 Cleaning steps"),
        bullet("Removed placeholder rows ('-' values) from geography and mode lookup tables"),
        bullet("Filtered ratings to the valid 1-5 range and months to 1-12"),
        bullet("Deduplicated all lookup and fact tables on their primary keys"),
        bullet("Filled missing User CityId with a sentinel value rather than dropping rows"),
        bullet("Joined geography (City → Country → Region → Continent) onto each user"),
        h2("2.2 Result"),
        p("52,930 transactions were retained after cleaning, covering 33,530 unique users and (in the transaction log) 30 distinct attractions actively visited, joined against the fuller 1,698-attraction catalog for content-based recommendations."),

        h1("3. Exploratory Data Analysis"),
        p("Key findings from the EDA:"),
        bullet("Average rating overall is 4.16/5 — ratings skew positive, with most falling in the 4-5 range"),
        bullet("'Couples' is the most common visit mode (41% of transactions), followed by Family and Friends"),
        bullet("Nature & Wildlife Areas is the most-visited attraction type; Water Parks earn the highest average rating among popular types"),
        bullet("The Americas account for the largest share of transactions by user continent"),

        image(`${FIG}/rating_distribution.png`),
        image(`${FIG}/visitmode_distribution.png`),
        image(`${FIG}/top_attraction_types.png`),
        image(`${FIG}/avg_rating_by_type.png`),
        image(`${FIG}/rating_by_visitmode.png`),
        image(`${FIG}/users_by_continent.png`),
        image(`${FIG}/visits_over_time.png`),
        image(`${FIG}/correlation_heatmap.png`),

        h1("4. Modeling"),
        h2("4.1 Regression — Predicting Rating"),
        p("A LightGBM regressor was trained on user geography, attraction type, visit timing, and user/attraction historical averages."),
        metricsTable([["R²", "0.746"], ["MSE", "0.239"], ["MAE", "0.259"]]),

        h2("4.2 Classification — Predicting Visit Mode"),
        p("A LightGBM classifier was trained to predict one of five visit modes (Business, Couples, Family, Friends, Solo), benchmarked against a Random Forest baseline."),
        metricsTable([
          ["LightGBM Accuracy", "0.508"], ["LightGBM F1 (weighted)", "0.462"],
          ["Random Forest Accuracy", "0.496"], ["Random Forest F1 (weighted)", "0.483"],
        ]),
        p("Performance is strongest for the majority classes (Couples, Family); minority classes (Business, Solo) are harder to predict due to class imbalance — a known limitation to flag for stakeholders."),

        h2("4.3 Recommendation System"),
        p("A truncated-SVD collaborative filtering model was built on the user-item rating matrix (mean-centered per user). A content-based fallback recommends attractions of the same type a user previously rated highly, used for cold-start users with no rating history."),
        metricsTable([["RMSE (held-out ratings)", "1.058"], ["Precision@5", "0.048"]]),
        p("The modest Precision@5 reflects the dataset's small attraction catalog (only 30 actively-visited attractions) relative to its user base, which limits differentiation between recommendations — a natural target for future improvement with more attraction inventory."),

        h1("5. Deployment"),
        p("All three models were wrapped in a multi-tab Streamlit application:"),
        bullet("Predict My Trip — enter geography, attraction type, and timing to get a predicted rating and visit mode"),
        bullet("Recommendations — select a user to see personalized, collaborative-filtering-based attraction suggestions"),
        bullet("Trends & Insights — an interactive Plotly dashboard of ratings, visit modes, attraction popularity, and continent-level traffic"),
        bullet("About — project and methodology summary"),
        p("The app was smoke-tested and confirmed to launch and serve requests successfully."),

        h1("6. Limitations & Next Steps"),
        bullet("Visit-mode classification accuracy (~51%) is moderate; richer behavioral features (trip duration, group size, prior mode history) could improve it"),
        bullet("The recommender is constrained by a small pool of actively-visited attractions; expanding real interaction data across the fuller 1,698-attraction catalog would improve personalization"),
        bullet("Regression performance benefits from attraction/user historical averages, which assumes some rating history exists — a pure cold-start rating predictor would need additional content features"),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("reports/Tourism_Experience_Analytics_Report.docx", buf);
  console.log("Report written.");
});
