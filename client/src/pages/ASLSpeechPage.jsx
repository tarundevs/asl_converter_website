import { ThemeToggle } from "../components/ThemeToggle.jsx";
import { Starbackground } from "@/components/Starbackground.jsx";
import { NavBar } from "@/components/NavBar.jsx";
import { ASL_speech } from "@/components/ASL_speech.jsx";

export const ASLSpeechPage = () => {
  return (
    <div className="min-h-screen bg-background text-foreground overflow-x-hidden">
      {/* Theme Toggle */}
      <ThemeToggle />
      {/* Background effects */}
      <Starbackground />
      {/* NavBar */}
      <NavBar />
      {/* Main Content */}
      <main>
        <ASL_speech />
      </main>
    </div>
  );
};