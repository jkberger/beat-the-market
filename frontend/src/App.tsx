import { Route, Routes } from "react-router-dom";
import { Layout } from "@/components/Layout";
import { BuilderPage } from "@/pages/BuilderPage";
import { HomePage } from "@/pages/HomePage";
import { ResultsPage } from "@/pages/ResultsPage";
import { StockPage } from "@/pages/StockPage";

export function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<HomePage />} />
        <Route path="stock/:symbol" element={<StockPage />} />
        <Route path="build" element={<BuilderPage />} />
        <Route path="results/:id" element={<ResultsPage />} />
      </Route>
    </Routes>
  );
}
