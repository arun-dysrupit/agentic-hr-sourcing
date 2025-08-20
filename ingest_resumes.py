import os
from dotenv import load_dotenv
from enhanced_resume_extractor import EnhancedResumeExtractor


def main() -> None:
    """Main function to ingest resumes using the enhanced LLM-based extractor."""
    load_dotenv()
    resume_dir = os.getenv("RESUME_DIR", "resumes")
    
    print("🚀 Enhanced Resume Ingestion System")
    print("=" * 50)
    print("This system uses advanced LLM analysis for accurate resume parsing")
    print("and stores enhanced candidate profiles in Couchbase.")
    print("=" * 50)
    
    # Check environment
    if not os.getenv("NEBIUS_API_KEY"):
        print("❌ NEBIUS_API_KEY not found in environment")
        print("   Please check your .env file")
        return
    
    try:
        # Initialize the enhanced resume extractor
        print("🔧 Initializing Enhanced Resume Extractor...")
        extractor = EnhancedResumeExtractor()
        
        # Process all resumes in the directory
        print(f"\n📁 Processing resumes from: {resume_dir}")
        processed_resumes = extractor.process_resumes_directory(resume_dir)
        
        # Display results
        print(f"\n🎉 Resume ingestion complete!")
        print(f"✅ Successfully processed {len(processed_resumes)} resumes")
        
        if processed_resumes:
            # Calculate statistics
            total_skills = sum(len(r.get("skills", [])) for r in processed_resumes)
            avg_confidence = sum(r.get("parsing_confidence", 0) for r in processed_resumes) / len(processed_resumes)
            
            print(f"\n📊 Ingestion Statistics:")
            print(f"   Total candidates: {len(processed_resumes)}")
            print(f"   Total skills extracted: {total_skills}")
            print(f"   Average parsing confidence: {avg_confidence:.1f}%")
            
            # Show confidence distribution
            high_confidence = sum(1 for r in processed_resumes if r.get("parsing_confidence", 0) >= 80)
            medium_confidence = sum(1 for r in processed_resumes if 50 <= r.get("parsing_confidence", 0) < 80)
            low_confidence = sum(1 for r in processed_resumes if r.get("parsing_confidence", 0) < 50)
            
            print(f"\n🎯 Parsing Confidence Distribution:")
            print(f"   High (≥80%): {high_confidence} resumes")
            print(f"   Medium (50-79%): {medium_confidence} resumes")
            print(f"   Low (<50%): {low_confidence} resumes")
            
            # Show top candidates by confidence
            top_candidates = sorted(processed_resumes, key=lambda x: x.get("parsing_confidence", 0), reverse=True)[:5]
            print(f"\n🏆 Top 5 Candidates by Parsing Confidence:")
            for i, candidate in enumerate(top_candidates, 1):
                print(f"   {i}. {candidate.get('name', 'Unknown')} - {candidate.get('parsing_confidence', 0):.1f}%")
                print(f"      Skills: {len(candidate.get('skills', []))}, Experience: {candidate.get('years_experience', 0)} years")
            
            # Show candidates that need attention
            low_confidence_candidates = [r for r in processed_resumes if r.get("parsing_confidence", 0) < 50]
            if low_confidence_candidates:
                print(f"\n⚠️  Candidates Needing Attention (Low Confidence):")
                for candidate in low_confidence_candidates:
                    print(f"   • {candidate.get('name', 'Unknown')} - {candidate.get('parsing_confidence', 0):.1f}%")
                    print(f"     File: {candidate.get('filename', 'Unknown')}")
            
            print(f"\n💾 All candidates have been stored in Couchbase")
            print(f"🔍 You can now use the enhanced search system to find candidates")
            
        else:
            print("❌ No resumes were processed successfully")
            
    except Exception as e:
        print(f"❌ Resume ingestion failed: {e}")
        print("Please check your environment configuration and try again.")


if __name__ == "__main__":
    main()

